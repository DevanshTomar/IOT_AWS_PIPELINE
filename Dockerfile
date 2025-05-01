FROM python:3.8-slim

# Set working directory for Lambda
WORKDIR /var/task

# Install only necessary system dependencies (with reduced size)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (smaller than GPU version)
RUN pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Setup cache directories
RUN mkdir -p /tmp/.cache
ENV TORCH_HOME=/tmp/.cache/torch
ENV XDG_CACHE_HOME=/tmp/.cache/torch

# Copy model weights and handlers
COPY resnetV1_video_weights.pt .
COPY resnetV1.pt .
COPY face-detection/fd_lambda.py .
COPY face-recognition/fr_lambda.py .

# Lambda runtime interface
ENTRYPOINT [ "python", "-m", "awslambdaric" ]

