#!/usr/bin/env bash
# Environment setup for PixelDiT_Segmentor (SAN + detectron2 + PixelDiT / Gemma2)
# Target: CUDA 13.x machines (e.g. RTX 4090 with CUDA 13.2 driver/toolkit).
# Combination: Python 3.11, torch 2.13.0 (CUDA 13.2 wheels, cu132), detectron2 from source,
# mmcv 1.7.2 (lite), open_clip_torch 2.16.0, protobuf 6.x, numpy < 2.
#
# Usage:  bash patched_install.sh
# Run from the repo root (the folder that contains SAN's requirements.txt).
#
# Overrides (environment variables):
#   CUDA_TAG=cu130|cu132|cu126   wheel flavour; auto-detected from nvcc / the driver if unset
#   TORCH_VERSION=2.13.0         torch release; torchvision is resolved to the matching release
#   GPU_ARCH=8.9                 compute capability to compile detectron2 for (8.9 = RTX 4090)

set -euo pipefail

# ---------------- settings ----------------
ENV_NAME="pixeldit"
PYTHON_VERSION="3.11"              # Pillow==9.3.0 (SAN reqs) has no wheels past 3.11
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
GPU_ARCH="${GPU_ARCH:-8.9}"
SAN_REQS="requirements.txt"
PIXELDIT_REQS=""                   # path to PixelDiT's requirements file, leave empty to skip
# ------------------------------------------

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

# 0. Work out which CUDA wheel flavour to use.
#    torch only refuses to build extensions when the nvcc MAJOR version differs from
#    torch.version.cuda, so a cu132 torch builds fine with nvcc 13.0-13.x.
detect_cuda_tag() {
  local ver=""
  if command -v nvcc >/dev/null 2>&1; then
    ver="$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    ver="$(nvidia-smi | sed -nE 's/.*CUDA Version: ([0-9]+\.[0-9]+).*/\1/p')"
  fi
  case "$ver" in
    13.0|13.1) echo "cu130" ;;
    13.*|1[4-9].*) echo "cu132" ;;
    12.*) echo "cu126" ;;
    *) echo "cu132" ;;
  esac
}
CUDA_TAG="${CUDA_TAG:-$(detect_cuda_tag)}"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
echo ">> Using torch ${TORCH_VERSION} from ${TORCH_INDEX}"

eval "$(conda shell.bash hook)"

# 1. Fresh env
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"

# torch >= 2.6 defaults torch.load(weights_only=True), which rejects the PixelDiT / detectron2
# checkpoints that pickle non-tensor objects. Restore the torch 2.4 behaviour for this env only.
conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
conda deactivate && conda activate "$ENV_NAME"

# 2. Torch first (everything else builds against it). torchvision comes from the same index,
#    so pip picks the release that matches TORCH_VERSION.
pip install "torch==${TORCH_VERSION}" torchvision --index-url "$TORCH_INDEX"
TORCHVISION_VERSION="$(python -c 'import torchvision; print(torchvision.__version__.split("+")[0])')"
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda)')"
echo ">> torch ${TORCH_VERSION} (CUDA ${TORCH_CUDA}), torchvision ${TORCHVISION_VERSION}"

# 3. Constraints that every later install must respect
cat > "$WORK_DIR/constraints.txt" <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
numpy<2
protobuf>=6.31.1,<7
opencv-python<5
mmcv==1.7.2
EOF

# 4. Build tools. setuptools<70 keeps pkg_resources, which mmcv's setup.py needs.
pip install 'numpy<2' 'setuptools<70' wheel ninja

# 5. nvcc whose major version matches torch's CUDA, needed to build detectron2.
TORCH_CUDA_MAJOR="${TORCH_CUDA%%.*}"
NVCC_MAJOR=""
if command -v nvcc >/dev/null 2>&1; then
  NVCC_MAJOR="$(nvcc --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p')"
fi
if [[ "$NVCC_MAJOR" != "$TORCH_CUDA_MAJOR" ]]; then
  echo ">> nvcc ${NVCC_MAJOR:-missing} does not match torch CUDA ${TORCH_CUDA}; installing CUDA ${TORCH_CUDA} toolkit into the env"
  conda install -y -c "nvidia/label/cuda-${TORCH_CUDA}.0" cuda-toolkit
  hash -r
fi
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
echo ">> Building with $(nvcc --version | tail -1) from ${CUDA_HOME}"

# 6. detectron2 from source, against the installed torch.
#    Compile only for the local GPU: faster, and CUDA 13 no longer builds for pre-Turing archs.
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
export FORCE_CUDA=1
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 7. SAN requirements, with the torch-1.x-era pins replaced
sed -E \
  -e 's/^mmcv==.*/mmcv==1.7.2/' \
  -e 's/^protobuf.*/protobuf>=6.31.1,<7/' \
  -e 's/^opencv-python([=<>].*)?$/opencv-python<5/' \
  "$SAN_REQS" > "$WORK_DIR/san_requirements_fixed.txt"
pip install --no-build-isolation -r "$WORK_DIR/san_requirements_fixed.txt" -c "$WORK_DIR/constraints.txt"

# If SAN code uses mmcv.ops (check: grep -rn "mmcv.ops" san/ t2i/), rebuild with ops:
# MMCV_WITH_OPS=1 pip install --no-build-isolation --force-reinstall --no-deps mmcv==1.7.2

# 8. PixelDiT requirements (transformers for Gemma2, pyrallis, etc.)
if [[ -n "$PIXELDIT_REQS" ]]; then
  pip install --no-build-isolation -r "$PIXELDIT_REQS" -c "$WORK_DIR/constraints.txt"
fi

# 9. Checks
python - <<'EOF'
import importlib.metadata as m
import torch, torchvision, numpy, cv2, mmcv, detectron2, wandb
from torch.nn.functional import scaled_dot_product_attention  # needs torch >= 2.0
from torch.utils.tensorboard import SummaryWriter
from detectron2 import _C  # compiled CUDA ops actually load
import open_clip
open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)

print("torch      ", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu        ", torch.cuda.get_device_name(0), "sm_%d%d" % torch.cuda.get_device_capability(0))
    x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()
print("torchvision", torchvision.__version__)
print("numpy      ", numpy.__version__)
print("opencv     ", cv2.__version__)
print("mmcv       ", mmcv.__version__)
print("detectron2 ", detectron2.__version__)
print("protobuf   ", m.version("protobuf"))
print("wandb      ", wandb.__version__)
print("open_clip  ", m.version("open_clip_torch"))
try:
    print("transformers", m.version("transformers"))
except m.PackageNotFoundError:
    print("transformers  NOT INSTALLED (needed for the Gemma2 text encoder)")
print("All imports OK")
EOF

echo
echo "Expected harmless warning: open-clip-torch 2.16.0 requires protobuf<4."
echo "Next: run a short training smoke test, e.g."
echo "  python train_net.py --config-file <cfg> --num-gpus 1 SOLVER.MAX_ITER 20 TEST.EVAL_PERIOD 20"