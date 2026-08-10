# MAI204 emotion recognition -- inference image
# Owner: Abdul Raouf Zabalawi
#
# Changes from the first draft, and why:
#   1. Installs requirements-api.txt (CPU torch, headless OpenCV) instead of the full
#      requirements.txt. The full file pulls mlflow, lime, grad-cam, evidently and the
#      CUDA torch wheel -- roughly 2.5 GB the service never imports. The checklist
#      requires the build to finish in under 5 minutes.
#   2. opencv-python-headless removes the libGL/X11 dependency chain entirely.
#      libgl1-mesa-glx is a transitional package on Debian bookworm and is gone in
#      trixie, so depending on it makes the build fragile against a base-image bump.
#      libgl1 + libglib2.0-0 stay as cheap insurance for the MediaPipe wheel.
#   3. libxrender-dev replaced with libxrender1 -- a runtime image should not carry
#      headers.
#   4. Runs as a non-root user.
#
# Build and run:
#   docker build -t mai204-emotion:latest .
#   docker run -p 5001:8000 -v "$(pwd)/results:/app/results" mai204-emotion:latest
#   curl http://localhost:5001/health
#
# The container starts without a checkpoint and reports model_loaded=false, so the CI
# smoke test can go green before the ablation finishes. Mount results/ to enable
# real inference.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so it stays cached unless requirements-api.txt changes
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY params.yaml .
COPY src/ ./src/

# Checkpoints and datasets are DVC-tracked -- never baked into the image
VOLUME ["/app/results", "/app/data"]

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
