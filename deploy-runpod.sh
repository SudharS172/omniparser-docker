#!/bin/bash

# RunPod Deployment Script for OmniParser API
# This script runs on RunPod startup to install dependencies and start the API

echo "🚀 Starting OmniParser API deployment on RunPod..."

# Update system and install dependencies
apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0

# Install Python dependencies
pip install fastapi[all] loguru ultralytics==8.3.81

# Clone and setup vendor dependencies  
if [ ! -d "/workspace/omniparser-api-v2" ]; then
    cd /workspace
    git clone https://github.com/lokkju/omniparser-api-v2.git
    cd omniparser-api-v2
    git submodule update --init --recursive
fi

cd /workspace/omniparser-api-v2

# Install omniparser dependencies
pip install -r vendor/omniparser/requirements.txt

# Create required directories
mkdir -p /workspace/imgs
mkdir -p /root/.cache/huggingface
mkdir -p /root/.config/matplotlib  
mkdir -p /root/.paddleocr
mkdir -p /root/.EasyOCR

# Set Python path
export PYTHONPATH=${PYTHONPATH}:/workspace/omniparser-api-v2/vendor/omniparser

echo "✅ Setup complete! Starting FastAPI server..."

# Start the API server
uvicorn app:app --host 0.0.0.0 --port 7860 