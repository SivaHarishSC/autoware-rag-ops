# Serving image: FastAPI + transformers + bitsandbytes, GPU inference.
# Base is plain python:3.12-slim, not an nvidia/cuda image -- the cu126
# torch wheel bundles its own CUDA runtime libs, it only needs a host
# with a compatible NVIDIA driver + nvidia-container-toolkit at `docker
# run` time (a K8s/host concern, not something this image needs to
# provide). Building this image needs no GPU -- it only installs
# Python packages; the model itself is pulled from the private HF repo
# at container *startup*, not at build time, so HF_TOKEN must be passed
# as a runtime env var (`docker run -e HF_TOKEN=...`), never baked in.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py model_config.py ./

EXPOSE 8000

CMD ["python", "server.py"]
