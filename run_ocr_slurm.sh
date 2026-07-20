#!/bin/bash
#SBATCH --job-name=ocr_fxo
#SBATCH --output=/mnt/slurm-beegfs/Users/j-vill36/scripts_ocr/logs/ocr_%j.out
#SBATCH --error=/mnt/slurm-beegfs/Users/j-vill36/scripts_ocr/logs/ocr_%j.err
#SBATCH --partition=cpu_express
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=computingjvm@gmail.com

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

BASE_DIR=/mnt/slurm-beegfs/Users/j-vill36/scripts_ocr
ENV_PATH="${BASE_DIR}/conda_envs"
PDF="FXO_SSRN_revised.pdf"

mkdir -p "${BASE_DIR}/logs" "${BASE_DIR}/papers"

# ==============================================================================
# Environment setup
# ==============================================================================
source /etc/profile.d/modules.sh 2>/dev/null || \
    source /usr/share/modules/init/bash 2>/dev/null || \
    source /opt/modules/init/bash 2>/dev/null

module load conda/3
eval "$(conda shell.bash hook)"

# Create env if not exists
if [ ! -f "$ENV_PATH/bin/python" ]; then
    echo ">>> Creating conda env (python=3.12)..."
    conda create --prefix "$ENV_PATH" python=3.12 pip --yes
fi

conda activate "$ENV_PATH"

# Install the package if not installed
if ! python -c "import paper_ocr" 2>/dev/null; then
    echo ">>> Installing paper-ocr..."
    pip install --no-cache-dir .
fi

echo ">>> Python: $(python --version)"
echo ">>> PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")')"

# Use local SSD for model cache (faster I/O than beegfs)
export HF_HOME=/tmp/hf_cache_${SLURM_JOB_ID}
export TORCH_HOME=/tmp/torch_cache_${SLURM_JOB_ID}
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ==============================================================================
# Run OCR
# ==============================================================================
cd "$BASE_DIR"
echo ">>> Running OCR on $PDF..."
ocr "$PDF" --force-ocr

echo ">>> OCR complete. Output at ${BASE_DIR}/papers/FXO_SSRN_revised/"
