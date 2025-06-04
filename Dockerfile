FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

USER root

# Install system dependencies
RUN apt update -q && apt install -y --no-install-recommends \
  git curl sudo wget unzip \
  libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
  build-essential gcc g++ make cmake ninja-build \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./vendor ./vendor

# PyTorch is already installed in base image - just upgrade if needed
RUN pip install --upgrade pip setuptools wheel

# Install core ML packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install transformers timm einops==0.8.0 numpy

# Install OCR and additional packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install easyocr paddlepaddle paddleocr opencv-python opencv-python-headless

# Install other vendor requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install supervision==0.18.0 openai==1.3.5 azure-identity gradio dill accelerate

# Install our additional packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fastapi[all] loguru ultralytics==8.3.81

# Fix transformers version for compatibility
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade "transformers>=4.35.0,<4.40.0"

# Install flash_attn for Florence2 model with proper build environment
ENV MAX_JOBS=1
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flash-attn --no-build-isolation

RUN mkdir -p /root/.cache/huggingface \
    && mkdir -p /root/.config/matplotlib \
    && mkdir -p /root/.paddleocr \
    && mkdir -p /root/.EasyOCR \
    && mkdir -p /app/imgs

ENV PYTHONPATH=/app/vendor/omniparser
COPY app.py app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
