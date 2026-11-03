#!/usr/bin/env bash
# Environment setup for PixelDiT_Segmentor (SAN + detectron2 + PixelDiT / Gemma2)
# Tested combination: Python 3.10, torch 2.4.1 (CUDA 12.1), detectron2 from source,
# mmcv 1.7.2 (lite), open_clip_torch 2.16.0, protobuf 6.x, numpy < 2.
#
# Usage:  bash setup_pixeldit_env.sh
# Run from the repo root (the folder that contains SAN's requirements.txt).

set -euo pipefail

# ---------------- settings ----------------
ENV_NAME="pixeldit"
PYTHON_VERSION="3.10"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"   # must match your CUDA driver / nvcc
SAN_REQS="requirements.txt"
PIXELDIT_REQS=""          # path to PixelDiT's requirements file, leave empty to skip
# ------------------------------------------

eval "$(conda shell.bash hook)"

# 1. Fresh env
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"

# 2. Torch first (everything else builds against it)
pip install torch==2.4.1 torchvision==0.19.1 --index-url "$TORCH_INDEX"

# 3. Constraints that every later install must respect
cat > /tmp/pixeldit_constraints.txt <<'EOF'
torch==2.4.1
torchvision==0.19.1
numpy<2
protobuf>=6.31.1,<7
opencv-python<5
mmcv==1.7.2
EOF

# 4. Build tools. setuptools<70 keeps pkg_resources, which mmcv's setup.py needs.
pip install 'numpy<2' 'setuptools<70' wheel ninja

# 5. detectron2 from source, against the installed torch
#    (needs nvcc matching torch.version.cuda; if missing:
#     conda install -c nvidia cuda-toolkit=12.1 -y)
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 6. SAN requirements, with the torch-1.x-era pins replaced
sed -E \
  -e 's/^mmcv==.*/mmcv==1.7.2/' \
  -e 's/^protobuf.*/protobuf>=6.31.1,<7/' \
  -e 's/^opencv-python([=<>].*)?$/opencv-python<5/' \
  "$SAN_REQS" > /tmp/san_requirements_fixed.txt
pip install --no-build-isolation -r /tmp/san_requirements_fixed.txt -c /tmp/pixeldit_constraints.txt

# If SAN code uses mmcv.ops (check: grep -rn "mmcv.ops" san/ t2i/), rebuild with ops:
# MMCV_WITH_OPS=1 pip install --no-build-isolation --force-reinstall --no-deps mmcv==1.7.2

# 7. PixelDiT requirements (transformers for Gemma2, pyrallis, etc.)
if [[ -n "$PIXELDIT_REQS" ]]; then
  pip install --no-build-isolation -r "$PIXELDIT_REQS" -c /tmp/pixeldit_constraints.txt
fi

# 8. Checks
python - <<'EOF'
import importlib.metadata as m
import torch, torchvision, numpy, cv2, mmcv, detectron2, wandb
from torch.nn.functional import scaled_dot_product_attention  # needs torch >= 2.0
from torch.utils.tensorboard import SummaryWriter
import open_clip
open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)

print("torch      ", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
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