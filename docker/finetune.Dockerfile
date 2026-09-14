# ltm-ft on an NVIDIA GPU: the same package and lockfile as the laptop runs, with the CUDA build of torch.
# Weights are NOT baked in: TabFM's checkpoint is downloaded from Hugging Face inside the pod (fast from inside GCP),
# which keeps the image free of the non-commercial weights.
#
#   Built by Cloud Build (k8s/gke/cloudbuild.yaml), natively on amd64. Locally, for a CUDA host:
#   docker buildx build --platform linux/amd64 -f docker/finetune.Dockerfile -t ltm-ft:cu126 .
FROM python:3.14-slim

ARG TORCH_VERSION=2.14.0
ARG TORCH_VARIANT=cu126
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app
# Dependency layer first, so code edits do not reinstall torch. requirements.txt is generated from uv.lock
# (docker/requirements.sh) without torch and the CUDA wheels; torch comes from the variant-specific PyTorch index.
COPY docker/requirements.txt ./requirements.txt
RUN uv pip install --system --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} torch==${TORCH_VERSION} \
    && uv pip install --system -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv pip install --system --no-deps .

# Non-root. Outputs and the Hugging Face cache live on emptyDir volumes mounted by the Job.
RUN useradd --uid 10001 --create-home app
USER 10001
ENV HF_HOME=/tmp/hf
WORKDIR /work
ENTRYPOINT ["ltm-ft"]
