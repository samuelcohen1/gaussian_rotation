#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=03:00:00
#SBATCH --account=eel5840
#SBATCH --qos=eel5840
#SBATCH --output=results/slurm-setup-%j.out
#SBATCH --error=results/slurm-setup-%j.err

# CPU-only. Builds .conda-env from environment.yml. Do not run this on a login node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results

module load conda
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -x .conda-env/bin/python ]; then
  conda env update -p .conda-env -f environment.yml
else
  conda env create -p .conda-env -f environment.yml
fi

.conda-env/bin/python -c "import torch, diffusers, PIL; print('env ok', torch.__version__)"
