#!/usr/bin/env bash
# Delete every billable resource up.sh created: the cluster (with its node pools), the image repository and the
# Cloud Build source uploads.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=k8s/gke/env.sh
source "$HERE/env.sh"

gcloud container clusters delete "$CLUSTER" --zone "$ZONE" --project "$PROJECT" --quiet || true
gcloud artifacts repositories delete "$REPO" --location "$REGION" --project "$PROJECT" --quiet || true
# Cloud Build source tarballs (gcloud builds submit creates <project>_cloudbuild on first use).
gcloud storage rm --recursive "gs://${PROJECT}_cloudbuild/source" --project "$PROJECT" || true
kubectl config delete-context "$CTX" 2>/dev/null || true
echo "remaining clusters:"; gcloud container clusters list --project "$PROJECT"
