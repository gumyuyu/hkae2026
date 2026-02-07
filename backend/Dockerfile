# Base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Step 1: Install system dependencies ---
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# --- Step 2: Install matching CUDA toolkit manually ---
RUN wget https://developer.download.nvidia.com/compute/cuda/12.1.105/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.105-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.105-1_amd64.deb && \
    cp /var/cuda-repo-12-1-local/7fa2af80.pub /usr/share/keyrings/ && \
    apt-key add /var/cuda-repo-12-1-local/7fa2af80.pub && \
    apt-get update && \
    apt-get install -y cuda && \
    rm -rf /var/lib/apt/lists/* cuda-repo-ubuntu2204-12-1-local_12.1.105-1_amd64.deb

# --- Step 3: Upgrade pip ---
RUN pip install --no-cache-dir --upgrade pip

# --- Step 4: Set working directory ---
WORKDIR /app

# --- Step 5: Copy all files ---
COPY . .

# --- Step 6: Install general Python requirements ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Step 7: Install Hunyuan3D dependencies ---
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# --- Step 8: Install Hunyuan3D inner layer ---
WORKDIR /app/inner_layer/models/hunyuan3D
RUN pip install --no-cache-dir -r hunyuan3d_requirements.txt
RUN pip install -e .

# --- Step 9: Texture & differentiable renderer setup ---
WORKDIR /app/inner_layer/models/hunyuan3D/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py install

WORKDIR /app/inner_layer/models/hunyuan3D/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py install

# --- Step 10: Back to app root ---
WORKDIR /app

# --- Step 11: Run FastAPI ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
