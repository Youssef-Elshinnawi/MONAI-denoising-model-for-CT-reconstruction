# CPU-only base image -- no institutional GPU test environment available to
# validate a CUDA variant locally. See the "Packaging as a Frontier
# Developer Portal processing unit" section of the repo root README.md.
FROM python:3.11-slim

WORKDIR /app

# Dependencies first, app code after -- so changing source files doesn't
# invalidate Docker's cached pip install layer on rebuild.
#
# torch installed separately via PyTorch's CPU-only wheel index: plain
# `pip install torch` defaults to the CUDA-enabled build (bundling ~1GB+ of
# NVIDIA driver libraries) even on a CPU-only base image -- wrong for a
# build we can only actually test on CPU hardware.
COPY frontier-app/requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 && \
    pip install -r requirements.txt

# Flat into /app (not a package) so entrypoint.py's plain `from inference
# import ...` / `from model import ...` resolve without PYTHONPATH tricks.
COPY src/model.py .
COPY src/inference.py .
COPY frontier-app/entrypoint.py .

# Both checkpoints included so config_schema.json's checkpoint_path choice
# is actually satisfiable at runtime -- matches entrypoint.py's
# DEFAULT_CHECKPOINT = "/app/checkpoints/...".
COPY checkpoints/best_model_run1_20ep.pt checkpoints/
COPY checkpoints/best_model_run2_100ep.pt checkpoints/

ENTRYPOINT ["python", "entrypoint.py"]
