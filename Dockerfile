# Heart Disease Risk API - serving image
# Build:  docker build -t heart-risk-api:1.0.0 .
# Run:    docker run -p 8000:8000 heart-risk-api:1.0.0
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

WORKDIR /app

# 1) Model first: install the EXACT scikit-learn version it was trained with
#    (read from the model card) so the pickled pipeline loads identically.
COPY models/model.joblib models/model_card.json ./models/
COPY requirements-api.txt .
RUN SKLEARN_VERSION=$(python -c "import json; print(json.load(open('models/model_card.json'))['sklearn'])") \
 && echo "Pinning scikit-learn==${SKLEARN_VERSION}" \
 && pip install "scikit-learn==${SKLEARN_VERSION}" \
 && pip install -r requirements-api.txt

# 2) Application code (only the package - no notebooks, data or tests)
COPY src/heart_mlops ./src/heart_mlops

# 3) Run as a non-root user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "heart_mlops.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
