# Shared DSGN++ Docker run settings (source from other dsgn2_*.sh scripts).
DSGN2_DOCKER_IMAGE="${DSGN2_DOCKER_IMAGE:-dsgn2:pt171}"
SPCONV_LIB_DIR="/opt/conda/lib/python3.8/site-packages/spconv"
DSGN2_GPU_FLAG=(--device nvidia.com/gpu=all)
# spconv wheels ship libcuhash.so beside libspconv.so; linker needs this dir.
DSGN2_LIB_ENV=(-e "LD_LIBRARY_PATH=${SPCONV_LIB_DIR}:/usr/local/cuda/lib64")
