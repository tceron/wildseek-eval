#!/bin/bash
#SBATCH --job-name=query_eval
#SBATCH --account=ceron
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=64000MB
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tanise.ceron@unibocconi.it
#SBATCH --output=./slurm_out/%x_%j.out
#SBATCH --error=./slurm_err/%x_%j.err

set -euo pipefail

CLASSIFICATION_TYPE="${CLASSIFICATION_TYPE:-highrisk}"
MODEL_PATH="${MODEL_PATH:-}"
PROMPTS_PATH="${PROMPTS_PATH:-}"
DATA_PATH="${DATA_PATH:-}"
FILE_PATH="${FILE_PATH:-}"
MAX_LEN="${MAX_LEN:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

jobname="query_eval_${CLASSIFICATION_TYPE}"

echo "Starting job: ${jobname}"
echo "Start time: $(date)"

hostname

module load miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate modernbert_env

mkdir -p slurm_out slurm_err

cmd=(python3 eval_modernbert.py --classification-type "${CLASSIFICATION_TYPE}" --max-len "${MAX_LEN}")

if [[ -n "${MODEL_PATH}" ]]; then
  cmd+=(--model-path "${MODEL_PATH}")
fi

if [[ -n "${PROMPTS_PATH}" ]]; then
  cmd+=(--prompts-path "${PROMPTS_PATH}")
fi

if [[ -n "${DATA_PATH}" ]]; then
  cmd+=(--data-path "${DATA_PATH}")
fi

if [[ -n "${FILE_PATH}" ]]; then
  cmd+=(--file-path "${FILE_PATH}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"

echo "--- Job Finished ---"
echo "End time: $(date)"
