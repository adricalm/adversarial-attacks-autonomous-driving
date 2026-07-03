# Arka-compatible DSGN inference image (PyTorch 1.3, Python 3.7).
# Build (host, needs sudo): bash scripts/dsgn_build_docker.sh
# Run: bash scripts/dsgn_run_inference_docker.sh
FROM pytorch/pytorch:1.3.0-cuda10.1-cudnn7-devel

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/DSGN_custom

# requirement.txt pins torch/torchvision — image already has CUDA build.
COPY external/DSGN_custom/requirement.txt /tmp/dsgn-requirements.txt
RUN pip install --no-cache-dir \
    "numpy<1.22" "scipy<1.8" "scikit-image<0.20" "numba<0.57" \
    yacs opencv-python fire tensorboardX "pillow<10" "imageio<2.31" Cython

# Source is bind-mounted at runtime; setup script builds extensions inside container.
