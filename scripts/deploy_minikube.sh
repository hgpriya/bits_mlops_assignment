#!/usr/bin/env bash
# Build the API image, load it into minikube and deploy the k8s manifests.
# Usage (from the project root):  bash scripts/deploy_minikube.sh
set -euo pipefail

IMAGE="heart-risk-api:1.0.0"

if ! command -v minikube >/dev/null 2>&1; then
  echo ">> installing minikube"
  curl -sLo /tmp/minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
  sudo install /tmp/minikube /usr/local/bin/minikube
fi
if ! command -v kubectl >/dev/null 2>&1; then
  echo ">> installing kubectl"
  curl -sLo /tmp/kubectl "https://dl.k8s.io/release/$(curl -sL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  sudo install /tmp/kubectl /usr/local/bin/kubectl
fi

if ! minikube status >/dev/null 2>&1; then
  echo ">> starting minikube"
  minikube start --driver=docker --cpus=2 --memory=4096
fi
minikube addons enable ingress >/dev/null 2>&1 || echo "(ingress addon not enabled - optional)"

echo ">> building ${IMAGE}"
docker build -t "${IMAGE}" .
echo ">> loading image into minikube"
minikube image load "${IMAGE}"

echo ">> applying manifests"
kubectl apply -k k8s/
kubectl -n heart-risk rollout restart deployment/heart-risk-api
kubectl -n heart-risk rollout status deployment/heart-risk-api --timeout=180s

kubectl -n heart-risk get deploy,pods,svc,ingress -o wide
echo
echo "Next: in a SECOND terminal run 'minikube tunnel' (keep it open), then:"
echo "  kubectl -n heart-risk get svc heart-risk-api   # note EXTERNAL-IP"
echo "  python scripts/smoke_test.py --url http://<EXTERNAL-IP>"
