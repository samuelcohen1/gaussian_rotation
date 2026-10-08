#!/bin/bash
#SBATCH --job-name=geom-corr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=06:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --account=eel5840
#SBATCH --qos=eel5840
#SBATCH --output=results/slurm-geometric-%j.out
#SBATCH --error=results/slurm-geometric-%j.err

# One GPU. Each sample generates one image and inverts it once per geometric condition.
# 100 samples x 13 conditions is a few hours. Change these three lines to rerun.
NUM_SAMPLES=100
SEED=42
OUTPUT_DIR="results/geometric_registration/run_001"

MODEL_ID="runwayml/stable-diffusion-v1-5"
NUM_INFERENCE_STEPS=50
DTYPE="float16"
IMAGE_SIZE=512
PROMPT="a photograph of an astronaut riding a horse"

# runwayml/stable-diffusion-v1-5 is gated. Accept the license and run
# `huggingface-cli login` once on HiPerGator before submitting.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results

export HF_HUB_CACHE="${SLURM_SUBMIT_DIR}/.cache/huggingface/hub"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [ ! -x .conda-env/bin/python ]; then
  echo "ERROR: .conda-env is missing. Do not install on the login node." >&2
  echo "  sbatch scripts/setup_env.slurm" >&2
  exit 1
fi

if ! .conda-env/bin/python -c "import torch, diffusers, PIL, scipy, pandas" >/dev/null 2>&1; then
  echo "ERROR: .conda-env cannot import torch, diffusers, PIL, scipy, and pandas." >&2
  echo "  sbatch scripts/setup_env.slurm" >&2
  exit 1
fi

.conda-env/bin/python scripts/run_geometric_registration_experiment.py \
  --num-samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-id "${MODEL_ID}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --device cuda \
  --dtype "${DTYPE}" \
  --image-size "${IMAGE_SIZE}" \
  --prompt "${PROMPT}"
