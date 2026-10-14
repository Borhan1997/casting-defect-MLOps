"""FastAPI service for the casting-defect classifier.

Run locally:
    uvicorn src.api:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from src.inference import load_model, predict


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


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)) -> PredictionResponse:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        result = predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a readable image.")

    return PredictionResponse(**result)