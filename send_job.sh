#!/bin/bash
set -euo pipefail

# Define model and prompt combinations
declare -a MODELS=(
  # "meta-llama/Llama-3.2-1B-Instruct"
  # "google/gemma-3-27b-it"
  # "meta-llama/Llama-3.1-70B-Instruct"
  # "Qwen/Qwen2.5-72B-Instruct"
  # "deepseek-ai/DeepSeek-V3"
  "Qwen/Qwen3-32B"
  # "allenai/Olmo-3.1-32B-Instruct"
)

# Create directories
mkdir -p jobs logs logs_err

# Loop through combinations and create individual job files
for model in "${MODELS[@]}"; do
  safe_model_name=$(echo "$model" | sed 's|/|-|g')  # Replace slashes for safe filenames
  jobname="job_${safe_model_name}"
  jobfile="jobs/${jobname}.sbatch"

    # Write job file (no blank line before the shebang!)
    cat > "$jobfile" <<EOF
#!/usr/bin/bash -l
#SBATCH --job-name=${jobname}
#SBATCH --account=ceron
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --mem=160000MB
#SBATCH --time=24:00:00   # Adjust as needed
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tanise.ceron@unibocconi.it
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs_err/%x_%j.err

echo "Starting job: ${jobname}"
echo "Start time: \$(date)"

hostname

module load miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

#!/bin/bash

# Activate the environment
# conda activate modernbert_env
conda activate bias-training-data

which python
python -V
pip list | grep transformers || true

echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Start time: \$(date)"

nvidia-smi

echo "Starting Python script..."
python3 prompt_models.py -m "$model" -lp "/data/milanlp/huggingface/hub"

echo "--- Job Finished ---"
echo "End time: \$(date)"
EOF

  # Submit the job
  sbatch "$jobfile"
  # echo "Would submit: $jobfile"
done
