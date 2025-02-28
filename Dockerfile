FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ARG OMNIPARSER_RELASE=v.2.0.0
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

USER root

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt update -q && apt install -y --no-install-recommends git curl python3 python3-pip sudo libgl1-mesa-glx libglib2.0-0

# download the repository
RUN curl -OL https://github.com/microsoft/OmniParser/archive/refs/tags/${OMNIPARSER_RELASE}.tar.gz \
    && tar -xvf ${OMNIPARSER_RELASE}.tar.gz \
    && rm ${OMNIPARSER_RELASE}.tar.gz \
    && mv OmniParser-${OMNIPARSER_RELASE} /opt/omniparser

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /opt/omniparser/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install fastapi[all] loguru

ENV PYTHONPATH=/opt/omniparser:$PYTHONPATH

RUN mkdir -p /root/.cache/huggingface \
    && mkdir -p /root/.config/matplotlib \
    && mkdir -p /root/.paddleocr \
    && mkdir -p /root/.EasyOCR

WORKDIR /app

COPY main.py main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
