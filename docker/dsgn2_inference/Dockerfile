# DSGN++ (DSGN2) inference image — PyTorch 1.7.1 + CUDA 11.0.
# Build (host, needs sudo): bash scripts/dsgn2_build_docker.sh
# Smoke test: bash scripts/dsgn2_smoke_test.sh
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0+PTX"
ENV FORCE_CUDA="1"
ENV SPCONV_FORCE_BUILD_CUDA="1"
ENV CUDA_HOME="/usr/local/cuda"
ENV MAX_JOBS=4

# Base image ships a stale NVIDIA CUDA apt repo (expired GPG key). CUDA is
# already in this devel image — only install build deps from Ubuntu archives.
RUN set -eux; \
    for f in /etc/apt/sources.list.d/*; do \
      [ -f "$f" ] && sed -i '/developer.download.nvidia.com\/compute\/cuda/d' "$f" || true; \
    done; \
    rm -f /etc/apt/sources.list.d/cuda.list 2>/dev/null || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build patchelf \
    libboost-dev libboost-filesystem-dev libboost-system-dev \
    libgl1-mesa-glx libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pin pip/setuptools for legacy packages (spconv 1.2.1, mmcv 1.4.0).
RUN pip install --no-cache-dir "pip<24" "setuptools<70" wheel

# pycocotools 2.0.2 breaks with Cython 3.x (wrong maskApi.c path in setup.py).
RUN pip install --no-cache-dir "cython<3"

# Base Python deps (DSGN2 / OpenPCDet stack).
RUN pip install --no-cache-dir \
    pycocotools==2.0.2 \
    torchvision==0.8.2 \
    numpy==1.19.5 \
    numba==0.48.0 \
    matplotlib==3.5.3 \
    tensorboardX \
    easydict \
    pyyaml \
    "scikit-image<0.20" \
    tqdm \
    SharedArray \
    opencv-python==4.5.5.64 \
    gdown

# spconv 1.2.1 — compile from source for sm_89 PTX JIT compatibility.
# Docker build has no GPU; SPCONV_FORCE_BUILD_CUDA forces CUDA compile anyway.
RUN pip install --no-cache-dir "cmake>=3.18,<4" && \
    git clone https://github.com/traveller59/spconv --recursive /tmp/spconv && \
    cd /tmp/spconv && \
    git reset --hard f22dd9 && \
    git submodule update --recursive && \
    python setup.py bdist_wheel && \
    pip install /tmp/spconv/dist/spconv-1.2.1-*.whl && \
    rm -rf /tmp/spconv

# libspconv.so loads libcuhash.so from the same package dir; set RPATH + env fallback.
RUN SPCONV_DIR=/opt/conda/lib/python3.8/site-packages/spconv && \
    patchelf --set-rpath '$ORIGIN' "${SPCONV_DIR}/libspconv.so" && \
    for so in "${SPCONV_DIR}"/spconv_utils*.so*; do \
      [ -f "$so" ] && patchelf --set-rpath '$ORIGIN' "$so" || true; \
    done

ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.8/site-packages/spconv:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# mmcv-full 1.4.0 — compile CUDA ops from source (prebuilt wheels lack sm_89).
# Use non-editable install so we can delete the clone (editable -e breaks after rm).
RUN git clone --depth 1 --branch v1.4.0 https://github.com/open-mmlab/mmcv.git /tmp/mmcv && \
    cd /tmp/mmcv && \
    MMCV_WITH_OPS=1 pip install --no-cache-dir . && \
    rm -rf /tmp/mmcv

# Clone DSGN++ repo and install bundled mmdetection + OpenPCDet (CUDA extensions).
RUN git clone https://github.com/chenyilun95/DSGN2.git /workspace/DSGN2

WORKDIR /workspace/DSGN2

RUN cd mmdetection-v2.22.0 && pip install -e .

# OpenPCDet CUDA ops (build_cost_volume, iou3d_nms, etc.) compile here.
RUN pip install -e .

WORKDIR /workspace/DSGN2
