# BDD100K Object Detection - Docker Image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Prevent tzdata from asking questions during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tzdata \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create necessary directories (empty; real data is mounted)
RUN mkdir -p \
    /data/mounted \
    /workspace/data/raw \
    /workspace/data/subset \
    /workspace/data/analysis \
    /workspace/data/evaluation \
    /workspace/data/training

# Copy and set up entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
