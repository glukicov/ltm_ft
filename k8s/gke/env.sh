# shellcheck shell=bash
# Shared settings for the GKE scripts. Source it; every value can be overridden from the environment.
# Your GCP project: $PROJECT, else the active gcloud project. It is never written into the repository.
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
export PROJECT="${PROJECT:?set PROJECT to your GCP project id (or run: gcloud config set project <id>)}"
export REGION="${REGION:-us-central1}"
export ZONE="${ZONE:-us-central1-a}"
export CLUSTER="${CLUSTER:-ltm-ft}"
export NAMESPACE="${NAMESPACE:-ltm-ft}"
export REPO="${REPO:-ltm-ft}"
export IMAGE="${IMAGE:-$REGION-docker.pkg.dev/$PROJECT/$REPO/ltm-ft:cu126}"
# A dedicated kubeconfig: these scripts never change the context your shell has selected.
export KUBECONFIG="${LTM_KUBECONFIG:-$HOME/.kube/ltm-ft}"
export CTX="gke_${PROJECT}_${ZONE}_${CLUSTER}"
# kubectl authenticates to GKE through gke-gcloud-auth-plugin, which ships in the gcloud SDK's bin/.
command -v gke-gcloud-auth-plugin >/dev/null 2>&1 ||
  PATH="$PATH:$(gcloud info --format='value(installation.sdk_root)' 2>/dev/null)/bin"
export PATH
