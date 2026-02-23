FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Python 3.12 via deadsnakes PPA (Ubuntu 22.04 only has 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    git git-lfs wget ffmpeg libsndfile1 curl ca-certificates \
    libgl1-mesa-glx libglib2.0-0 poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Fix blinker conflict (dpkg not apt remove!)
RUN dpkg --force-depends -r python3-blinker 2>/dev/null || true

WORKDIR /app

# PyTorch + CUDA
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# flash-attn: prebuilt wheel → ninja fallback → skip
RUN pip3 install --no-cache-dir ninja packaging psutil && \
    pip3 install --no-cache-dir \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    2>/dev/null || \
    (echo "Prebuilt failed, ninja build..." && \
     MAX_JOBS=8 pip3 install --no-cache-dir flash-attn --no-build-isolation) || \
    echo "flash-attn skipped — using PyTorch native SDPA"

# SAM3 model library + project deps
RUN pip3 install --no-cache-dir \
    sam3 modelscope \
    fastmcp pdf2image onnxruntime-gpu pyyaml \
    scikit-image opencv-python-headless Pillow numpy requests \
    fastapi "uvicorn[standard]" httpx cffi

# Download SAM3 model from ModelScope (no auth needed, ~3.2GB)
RUN python3 -c "\
from modelscope import snapshot_download; \
d = snapshot_download('facebook/sam3', cache_dir='/models/modelscope'); \
print(f'SAM3 model at: {d}')"

# Download BPE vocab (from OpenAI CLIP)
RUN mkdir -p /models/assets && \
    wget -q -O /models/assets/bpe_simple_vocab_16e6.txt.gz \
    "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"

# Copy project
COPY . .

RUN mkdir -p /app/input /app/output /tmp/edit-banana

EXPOSE 8450 8451 8452

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8450/health')"

CMD ["bash", "start.sh"]
