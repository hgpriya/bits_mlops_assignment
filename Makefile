# Shortcuts for every pipeline stage.  Run `make help` to list them.
PY ?= python
IMAGE ?= heart-risk-api:1.0.0

.PHONY: help setup data eda train test lint predict serve docker-build docker-run smoke k8s-deploy monitor-up monitor-down traffic mlflow-ui

help:  ## list targets
	@grep -E "^[a-z-]+:.*##" Makefile | sed "s/:.*##/ -/"
setup:  ## install all dependencies (runtime + dev)
	$(PY) -m pip install -r requirements-dev.txt
data:  ## download the raw UCI dataset
	$(PY) data/download_data.py
eda:  ## clean data + EDA figures
	$(PY) scripts/run_eda.py
train:  ## train, tune, track in MLflow, save model
	$(PY) scripts/train.py
test:  ## unit tests with coverage
	$(PY) -m pytest -v --cov=heart_mlops --cov-report=term-missing
lint:  ## flake8
	$(PY) -m flake8 src scripts tests data
predict:  ## CLI inference on the sample request
	$(PY) scripts/predict.py --input sample_request.json
serve:  ## run the API locally without Docker
	PYTHONPATH=src $(PY) -m uvicorn heart_mlops.api:app --host 0.0.0.0 --port 8000 --reload
mlflow-ui:  ## MLflow dashboard on port 5000
	mlflow ui --backend-store-uri sqlite:///$(CURDIR)/mlflow.db --port 5000
docker-build:  ## build the serving image
	docker build -t $(IMAGE) .
docker-run:  ## run the serving image on port 8000
	docker run --rm -d --name heart-risk-api -p 8000:8000 $(IMAGE)
smoke:  ## smoke-test a running API (URL=http://...)
	$(PY) scripts/smoke_test.py --url $(or $(URL),http://localhost:8000)
k8s-deploy:  ## deploy to minikube
	bash scripts/deploy_minikube.sh
monitor-up:  ## API + Prometheus + Grafana
	docker compose -f monitoring/docker-compose.yml up -d --build
monitor-down:  ## stop the monitoring stack
	docker compose -f monitoring/docker-compose.yml down
traffic:  ## send demo traffic to the API
	$(PY) scripts/send_requests.py --url $(or $(URL),http://localhost:8000) --n 300
