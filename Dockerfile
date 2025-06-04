FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update -q && apt install -y --no-install-recommends git curl python3 python3-pip sudo libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY ./vendor ./vendor

# Install vendor requirements first
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r vendor/omniparser/requirements.txt

# Install additional packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fastapi[all] loguru ultralytics==8.3.81

# Fix transformers version for compatibility
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade "transformers>=4.35.0,<4.40.0"

RUN mkdir -p /root/.cache/huggingface \
    && mkdir -p /root/.config/matplotlib \
    && mkdir -p /root/.paddleocr \
    && mkdir -p /root/.EasyOCR \
    && mkdir -p /app/imgs

ENV PYTHONPATH=${PYTHONPATH}:/app/vendor/omniparser
COPY app.py app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
