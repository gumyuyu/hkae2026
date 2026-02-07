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
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# --- Step 2: Set working directory ---
WORKDIR /app

# --- Step 3: Copy all files ---
COPY . .



# --- Step 4: Install other general requirements ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Step 5: Install Hunyuan3D dependencies ---
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# --- Step 6: Install Hunyuan3D dependencies ---
WORKDIR /app/inner_layer/models/hunyuan3D
RUN pip install --no-cache-dir -r hunyuan3d_requirements.txt

# --- Step 7: Texture & differentiable renderer setup ---
WORKDIR /app/inner_layer/models/hunyuan3D/hy3dpaint/custom_rasterizer
RUN pip install -e .

WORKDIR /app/inner_layer/models/hunyuan3D/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh

# --- Step 8: Back to app root ---
WORKDIR /app

# --- Step 9: Run FastAPI ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
