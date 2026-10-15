"""FastAPI service for the casting-defect classifier.

Run locally:
    uvicorn src.api:app --reload
Interactive docs: http://127.0.0.1:8000/docs
Prometheus metrics: http://127.0.0.1:8000/metrics
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from PIL import UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.inference import load_model, predict
from src.metrics import record_prediction
from src.monitoring import log_prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup: a missing or corrupt baseline.pt should
    # make the service fail at launch, not on the first request.
    load_model()
    yield


app = FastAPI(
    title="Casting defect classifier",
    description="Upload an image of a casting product; get back a label and P(defect).",
    version="0.1.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    label: str  # "def_front" or "ok_front"
    probability: float  # P(defect), between 0 and 1


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    # A plain route rather than a mounted ASGI app: a mount would redirect
    # GET /metrics to /metrics/.
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    start = time.perf_counter()
    try:
        result = predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a readable image.")
    latency = time.perf_counter() - start

    # Monitoring only runs for served predictions. Neither call can raise, so
    # neither can affect the response. Logging is opt-in (PREDICTION_LOG_PATH).
    record_prediction(image_bytes, result, latency)
    log_prediction(image_bytes, result)

    return PredictionResponse(**result)