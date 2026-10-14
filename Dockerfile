FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first: this layer is cached until requirements-serving.txt changes.
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Non-root user for the service.
RUN useradd --create-home --uid 1000 appuser

# Only what serving needs, not the training code.
# The package marker is created here so the image doesn't depend on the contents
# of the repo's src/__init__.py.
COPY src/api.py src/inference.py ./src/
RUN touch src/__init__.py && chown -R appuser:appuser /app

COPY --chown=appuser:appuser models/baseline.pt ./models/baseline.pt
ENV MODEL_PATH=/app/models/baseline.pt

USER appuser

EXPOSE 8000

# 0.0.0.0 so the port is reachable from outside the container.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]