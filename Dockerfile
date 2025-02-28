FROM registry.hf.space/microsoft-omniparser-v2:latest

USER root

RUN chmod 1777 /tmp \
    && apt update -q && apt install -y ca-certificates wget libgl1 \
    && wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb && apt update -q \
    && apt install -y --no-install-recommends libcudnn8 libcublas-12-2


USER user

RUN pip install fastapi[all]


COPY main.py main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
