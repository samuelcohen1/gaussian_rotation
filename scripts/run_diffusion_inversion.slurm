#!/bin/bash
#SBATCH --job-name=diff-invert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# 32gb was rejected with QOSMaxMemoryPerUser on the eel5840 QOS.
#SBATCH --mem=8gb
#SBATCH --time=02:00:00
# The old "gpu" partition was removed. hpg-turin is the L4 partition (24 GB).
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --account=eel5840
#SBATCH --qos=eel5840
#SBATCH --output=results/slurm-diffusion-%j.out
#SBATCH --error=results/slurm-diffusion-%j.err

# One GPU, one image, one inversion. Change the three lines below to rerun.
MODEL_ID="runwayml/stable-diffusion-v1-5"
SEED=12345
OUTPUT_DIR="results/diffusion_inversion/run_001"

NUM_INFERENCE_STEPS=50
DTYPE="float16"
IMAGE_SIZE=512
PROMPT="a photograph of an astronaut riding a horse"

# runwayml/stable-diffusion-v1-5 is a gated Hugging Face repo. Accept the license
# on the model page and run `huggingface-cli login` once on HiPerGator before
# submitting. The token stays in your home directory; model files go to the cache below.

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

if ! .conda-env/bin/python -c "import torch, diffusers, PIL" >/dev/null 2>&1; then
  echo "ERROR: .conda-env cannot import torch, diffusers, and PIL." >&2
  echo "  sbatch scripts/setup_env.slurm" >&2
  exit 1
fi

.conda-env/bin/python experiments/generate_and_invert.py \
  --model-id "${MODEL_ID}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --device cuda \
  --dtype "${DTYPE}" \
  --image-size "${IMAGE_SIZE}" \
  --prompt "${PROMPT}"
