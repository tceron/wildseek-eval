#!/bin/bash
#SBATCH --job-name=query_train
#SBATCH --account=ceron
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64000MB
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tanise.ceron@unibocconi.it
#SBATCH --output=./slurm_out/%x_%j.out
#SBATCH --error=./slurm_err/%x_%j.err

set -euo pipefail

CLASSIFICATION_TYPE="${CLASSIFICATION_TYPE:-highrisk}"
PROMPTS_PATH="${PROMPTS_PATH:-}"
PATH_SAVE_MODEL="${PATH_SAVE_MODEL:-}"
MODEL_SIZES="${MODEL_SIZES:-}"
MAX_LENS="${MAX_LENS:-}"

jobname="query_train_${CLASSIFICATION_TYPE}"

echo "Starting job: ${jobname}"
echo "Start time: $(date)"

hostname

module load miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate modernbert_env

mkdir -p slurm_out slurm_err

cmd=(python3 train_modernbert.py --classification-type "${CLASSIFICATION_TYPE}")

if [[ -n "${PROMPTS_PATH}" ]]; then
  cmd+=(--prompts-path "${PROMPTS_PATH}")
fi

if [[ -n "${PATH_SAVE_MODEL}" ]]; then
  cmd+=(--path-save-model "${PATH_SAVE_MODEL}")
fi

if [[ -n "${MODEL_SIZES}" ]]; then
  cmd+=(--model-sizes "${MODEL_SIZES}")
fi

if [[ -n "${MAX_LENS}" ]]; then
  cmd+=(--max-lens "${MAX_LENS}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "--- Job Finished ---"
echo "End time: $(date)"
