#!/bin/bash

# Define model and prompt combinations
declare -a DATASETS=(
  "en_ShareGPT"
  # "en_elisa"
  "en_WildChat"
  "en_lmsys-chat-1m"
)

# declare -a PROMPTS=("stance_1")  #"stance_2"

# Create directories
mkdir -p jobs logs logs_err

# Loop through combinations and create individual job files
for dataset in "${DATASETS[@]}"; do
  # for prompt in "${PROMPTS[@]}"; do
  safe_model_name=$(echo "$dataset" | sed 's|/|-|g')  # Replace slashes for safe filenames
  jobname="job_${safe_model_name}"
  jobfile="jobs/${jobname}.sbatch"

  cat > "$jobfile" <<EOF
#!/usr/bin/bash -l
#SBATCH --job-name=${jobname}
#SBATCH --account=ceron
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs_err/%x_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
#SBATCH --mail-user=tanise.ceron@unibocconi.it
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=10:00:00   # You can adjust this per job

echo "Starting job: $jobname"
hostname

module load miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frames

which python
python -V
pip list | grep transformers

echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Start time: \$(date)"

nvidia-smi

echo "Starting Python script..."
python3 classification_with_model_prompting.py -m "google/gemma-3-27b-it" -p "zero2_v4" -d "$dataset"

echo "--- Job Finished ---"
echo "End time: \$(date)"
EOF

  # Submit the job
  sbatch "$jobfile"
  # echo "Would submit: $jobfile"
done