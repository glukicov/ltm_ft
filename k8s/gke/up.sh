#!/usr/bin/env bash
# GKE for fine-tuning: a zonal Standard cluster with one small system node and an NVIDIA L4 node pool that autoscales
# from 0, so GPU cost exists only while a fine-tuning pod is scheduled. Builds the CUDA image with Cloud Build.
# `down.sh` deletes everything this creates.
#
#   k8s/gke/up.sh                  # cluster + image (idempotent: skips what exists)
#   REBUILD=1 k8s/gke/up.sh        # rebuild the image after code changes
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=k8s/gke/env.sh
source "$HERE/env.sh"

gcloud services enable cloudbuild.googleapis.com container.googleapis.com artifactregistry.googleapis.com \
  --project "$PROJECT"

gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION" --project "$PROJECT"

if [[ -n "${REBUILD:-}" ]] || ! gcloud artifacts docker images describe "$IMAGE" >/dev/null 2>&1; then
  gcloud builds submit "$HERE/../.." --project "$PROJECT" --config "$HERE/cloudbuild.yaml" \
    --substitutions "_REGION=$REGION,_REPO=$REPO"
fi

if ! gcloud container clusters describe "$CLUSTER" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  # Image streaming lazily pulls layers from Artifact Registry, so the multi-GB CUDA image starts in seconds.
  gcloud container clusters create "$CLUSTER" --project "$PROJECT" --zone "$ZONE" \
    --release-channel regular --num-nodes 1 --machine-type e2-standard-4 --enable-image-streaming
  # g2-standard-8: 1x L4 (24 GB), 8 vCPUs, 32 GB RAM. The taint keeps everything but GPU pods off it.
  gcloud container node-pools create l4 --project "$PROJECT" --cluster "$CLUSTER" --zone "$ZONE" \
    --machine-type g2-standard-8 --accelerator type=nvidia-l4,count=1,gpu-driver-version=latest \
    --enable-autoscaling --num-nodes 0 --min-nodes 0 --max-nodes 1 \
    --node-taints nvidia.com/gpu=present:NoSchedule --enable-image-streaming
fi
gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE" --project "$PROJECT"
kubectl --context "$CTX" create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl --context "$CTX" apply -f -
echo "GKE ready: k8s/gke/kctl get nodes"
