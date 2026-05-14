#!/usr/bin/env bash
# building conda 环境创建脚本
# CUDA 11.8 + PyTorch 2.x + 遥感建筑分割全套依赖
set -e

ENV_NAME="building"
PYTHON_VER="3.10"
LOG_PREFIX="[setup_env]"

echo "${LOG_PREFIX} ===== 开始创建 conda 环境 ====="
echo "${LOG_PREFIX} 环境名: ${ENV_NAME}  Python: ${PYTHON_VER}"

# 初始化 conda
source /root/miniconda3/etc/profile.d/conda.sh

# ── Step 1: 创建环境 ──────────────────────────────────────────
echo ""
echo "${LOG_PREFIX} [1/4] 创建 conda 环境..."
conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y
conda activate "${ENV_NAME}"
echo "${LOG_PREFIX} Python: $(python --version)"

# ── Step 2: 安装 PyTorch (CUDA 11.8) ─────────────────────────
echo ""
echo "${LOG_PREFIX} [2/4] 安装 PyTorch 2.x + CUDA 11.8..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ── Step 3: 安装其他依赖 ─────────────────────────────────────
echo ""
echo "${LOG_PREFIX} [3/4] 安装其他依赖..."
pip install \
    numpy \
    pandas \
    opencv-python \
    pillow \
    tifffile \
    scikit-image \
    scikit-learn \
    albumentations \
    matplotlib \
    tqdm \
    pyyaml \
    tensorboard \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── Step 4: 输出环境信息 ─────────────────────────────────────
echo ""
echo "${LOG_PREFIX} [4/4] 环境信息核查..."
python - <<'PYEOF'
import sys, torch, torchvision, cv2, numpy, pandas, PIL, tifffile, sklearn, albumentations

print("=" * 55)
print(f"  Python       : {sys.version.split()[0]}")
print(f"  PyTorch      : {torch.__version__}")
print(f"  torchvision  : {torchvision.__version__}")
print(f"  CUDA 可用    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本    : {torch.version.cuda}")
    print(f"  GPU 名称     : {torch.cuda.get_device_name(0)}")
    print(f"  GPU 显存     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"  NumPy        : {numpy.__version__}")
print(f"  OpenCV       : {cv2.__version__}")
print(f"  Pandas       : {pandas.__version__}")
print(f"  Pillow       : {PIL.__version__}")
print(f"  tifffile     : {tifffile.__version__}")
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  albumentations: {albumentations.__version__}")
print("=" * 55)
PYEOF

echo ""
echo "${LOG_PREFIX} ===== 安装完成 ====="
