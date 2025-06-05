# Use devel image (has nvcc for flash-attention compilation)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

USER root

# Set timezone non-interactively to avoid hanging
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH

# Install build dependencies, packages, and clean up in minimal layers
RUN apt update -q && apt install -y --no-install-recommends \
  git curl wget build-essential gcc g++ cmake \
  libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
  && pip install --no-cache-dir --upgrade pip setuptools wheel \
  && pip install --no-cache-dir \
     transformers timm einops==0.8.0 numpy \
     easyocr paddlepaddle paddleocr opencv-python opencv-python-headless \
     supervision==0.18.0 openai==1.3.5 azure-identity gradio dill accelerate \
     fastapi[all] loguru ultralytics==8.3.81 "transformers>=4.35.0,<4.40.0" \
  && pip install --no-cache-dir flash-attn==2.3.6 --no-build-isolation \
  && apt remove -y build-essential gcc g++ cmake \
  && apt autoremove -y && apt clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/pip

WORKDIR /app

# Copy vendor directory for reference and weights
COPY ./vendor ./vendor

# Copy utils.py, util directory, and app.py directly
COPY utils.py utils.py
COPY util util
COPY app.py app.py

RUN mkdir -p /root/.cache/huggingface /root/.config/matplotlib \
    /root/.paddleocr /root/.EasyOCR /app/imgs

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
