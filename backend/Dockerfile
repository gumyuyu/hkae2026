# ------------------------------
# Base image: Ubuntu 22.04 + CUDA 12.8 toolkit
# Mirrors your local setup: CUDA 12.8.93 + nvcc
# ------------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/usr/local/cuda/bin:${PATH}

# ------------------------------
# Step 1: Install system dependencies
# ------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    wget \
    ca-certificates \
    curl \
    gnupg \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Step 2: Upgrade pip
# ------------------------------
RUN python3 -m pip install --no-cache-dir --upgrade pip

# ------------------------------
# Step 3: Set working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# Step 4: Copy your project files
# ------------------------------
COPY . .

# ------------------------------
# Step 5: Install Python requirements
# ------------------------------
RUN pip install --no-cache-dir -r requirements.txt


# ------------------------------
# Step 6: Back to app root
# ------------------------------
WORKDIR /app

# ------------------------------
# Step 7: Run FastAPI
# ------------------------------
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]