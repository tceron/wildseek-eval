#!/bin/bash
set -euo pipefail

# Input CSV to evaluate (columns: prompt_id, query, domain, response)
INPUT="data_prompts/agreement_annotation_by_criterion_annotation_round2.csv"

# Define evaluator models to run
declare -a MODELS=(
  "Qwen/Qwen3.6-27B"
)

# Create directories
mkdir -p jobs logs logs_err

for model in "${MODELS[@]}"; do
  safe_model_name=$(echo "$model" | sed 's|/|-|g')
  safe_input=$(basename "$INPUT" .csv)
  jobname="safety_eval_${safe_model_name}_${safe_input}"
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
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tanise.ceron@unibocconi.it
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs_err/%x_%j.err

echo "Starting job: ${jobname}"
echo "Start time: \$(date)"

hostname

module load miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

conda activate safety_env

which python
python -V

echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_JOB_NODELIST"

nvidia-smi

echo "Starting Python script..."
python3 llm_safety_evaluator.py \
  --input "$INPUT" \
  --backend hf \
  --model "$model" \
  --local-path "/data/milanlp/huggingface/hub"

echo "--- Job Finished ---"
echo "End time: \$(date)"
EOF

  sbatch "$jobfile"
  echo "Submitted: $jobfile"
done
