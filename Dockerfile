FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update -q && apt install -y --no-install-recommends git curl python3 python3-pip sudo libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY ./vendor ./vendor

# Install fixed requirements first
COPY requirements-fix.txt requirements-fix.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-fix.txt

# Then install vendor requirements (may override some versions)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r vendor/omniparser/requirements.txt

RUN mkdir -p /root/.cache/huggingface \
    && mkdir -p /root/.config/matplotlib \
    && mkdir -p /root/.paddleocr \
    && mkdir -p /root/.EasyOCR \
    && mkdir -p /app/imgs

ENV PYTHONPATH=${PYTHONPATH}:/app/vendor/omniparser
COPY app.py app.py

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
