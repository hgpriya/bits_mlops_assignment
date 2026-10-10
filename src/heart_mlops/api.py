"""FastAPI model-serving app with request logging and Prometheus metrics.

Run locally (from the project root):
    PYTHONPATH=src uvicorn heart_mlops.api:app --port 8000

Endpoints
    GET  /health        liveness/readiness probe
    GET  /model-info    model card (name, params, metrics, versions)
    POST /predict       one patient  -> prediction + confidence
    POST /predict/batch up to 100 patients
    GET  /metrics       Prometheus metrics
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from heart_mlops import __version__, config
from heart_mlops.predict import load_model, load_model_card, predict_records


# ------------------------------------------------------------------ logging
class JsonFormatter(logging.Formatter):
    """One JSON object per line -> easy to grep, ship to ELK/Loki, or read in kubectl logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("heart_api")
    if logger.handlers:
        return logger
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    log_file = os.getenv("LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    logger.propagate = False
    return logger


log = _configure_logging()


# ------------------------------------------------------------------ schemas
class PatientFeatures(BaseModel):
    """Raw patient fields (UCI Cleveland coding). Ranges reject obvious typos."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 150, "chol": 270, "fbs": 0,
        "restecg": 2, "thalach": 111, "exang": 1, "oldpeak": 0.8, "slope": 1,
        "ca": 0, "thal": 7,
    }})

    age: int = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=70, le=220, description="Resting BP, mm Hg")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=60, le=220, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST")
    ca: Literal[0, 1, 2, 3] = Field(..., description="Major vessels coloured")
    thal: Literal[3, 6, 7] = Field(..., description="3 normal, 6 fixed, 7 reversible")


class PredictionResponse(BaseModel):
    request_id: str
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    risk_band: str
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    request_id: str
    count: int
    results: list[dict]


# ------------------------------------------------------------------ app
# Bounded label set for metrics (avoid one time-series per random URL).
KNOWN_PATHS = {"/", "/health", "/model-info", "/predict", "/predict/batch", "/metrics",
               "/docs", "/openapi.json"}


def create_app(model_path: str | os.PathLike | None = None) -> FastAPI:
    """App factory: a fresh metrics registry per app keeps tests isolated."""
    registry = CollectorRegistry()
    m_requests = Counter("api_requests_total", "HTTP requests",
                         ["method", "endpoint", "status"], registry=registry)
    m_latency = Histogram("api_request_latency_seconds", "Request latency",
                          ["endpoint"], registry=registry,
                          buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
    m_predictions = Counter("model_predictions_total", "Predictions by class",
                            ["predicted_class", "risk_band"], registry=registry)
    m_proba = Histogram("model_probability_disease", "Predicted disease probability",
                        registry=registry, buckets=[i / 10 for i in range(1, 11)])
    m_loaded = Gauge("model_loaded", "1 if the model is loaded", registry=registry)

    state: dict = {"model": None, "card": {}}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        path = Path(model_path or os.getenv("MODEL_PATH", config.MODEL_PATH))
        try:
            state["model"] = load_model(path)
            state["card"] = load_model_card(path.with_name("model_card.json"))
            m_loaded.set(1)
            log.info("model loaded", extra={"extra_fields": {
                "model_path": str(path), "model_name": state["card"].get("model_name")}})
        except Exception as exc:  # keep serving /health so the probe reports it
            m_loaded.set(0)
            log.error("model load failed", extra={"extra_fields": {"error": str(exc)}})
        yield

    app = FastAPI(title="Heart Disease Risk API", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            endpoint = getattr(route, "path", None) or (
                request.url.path if request.url.path in KNOWN_PATHS else "other")
            m_requests.labels(request.method, endpoint, str(status)).inc()
            m_latency.labels(endpoint).observe(elapsed)
            if endpoint != "/metrics":
                log.info("request", extra={"extra_fields": {
                    "request_id": request_id, "method": request.method,
                    "path": request.url.path, "status": status,
                    "latency_ms": round(elapsed * 1000, 2),
                    "client": request.client.host if request.client else None,
                }})

    def _require_model():
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return state["model"]

    def _score(patients: list[PatientFeatures]) -> list[dict]:
        model = _require_model()
        threshold = float(state["card"].get("decision_threshold", 0.5))
        results = predict_records(model, [p.model_dump() for p in patients], threshold)
        for r in results:
            m_predictions.labels(str(r["prediction"]), r["risk_band"]).inc()
            m_proba.observe(r["probability_disease"])
        return results

    @app.get("/")
    def root():
        return {"service": "heart-disease-risk-api", "docs": "/docs", "version": __version__}

    @app.get("/health")
    def health():
        loaded = state["model"] is not None
        body = {"status": "ok" if loaded else "degraded", "model_loaded": loaded}
        return Response(json.dumps(body), status_code=200 if loaded else 503,
                        media_type="application/json")

    @app.get("/model-info")
    def model_info():
        _require_model()
        return state["card"]

    @app.post("/predict", response_model=PredictionResponse)
    def predict(patient: PatientFeatures, request: Request):
        result = _score([patient])[0]
        log.info("prediction", extra={"extra_fields": {
            "request_id": request.state.request_id, **result}})
        return PredictionResponse(
            request_id=request.state.request_id,
            model_name=state["card"].get("model_name", "unknown"),
            model_version=state["card"].get("trained_at_utc", "unknown"),
            **result,
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request):
        results = _score(batch.patients)
        return BatchResponse(request_id=request.state.request_id,
                             count=len(results), results=results)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
