#!/bin/bash
set -euo pipefail

# Define models to run
declare -a MODELS=(
  # "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen3.6-27B"
)

# Create directories
mkdir -p jobs logs logs_err

for model in "${MODELS[@]}"; do
  safe_model_name=$(echo "$model" | sed 's|/|-|g')
  jobname="highrisk_${safe_model_name}"
  jobfile="jobs/${jobname}.sbatch"

  cat > "$jobfile" <<EOF
#!/usr/bin/bash -l
#SBATCH --job-name=${jobname}
#SBATCH --account=ceron
#SBATCH --partition=gpunew
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=160000MB
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tanise.ceron@unibocconi.it
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs_err/%x_%j.err

echo "Starting job: ${jobname}"
echo "Start time: \$(date)"

hostname

module load miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

conda activate bias-training-data

which python
python -V

echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"

nvidia-smi

echo "Starting Python script..."
python3 prompt_highrisk_models.py -m "$model" -lp "/data/milanlp/huggingface/hub"

echo "--- Job Finished ---"
echo "End time: \$(date)"
EOF

  sbatch "$jobfile"
  echo "Submitted: $jobfile"
done
