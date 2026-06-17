#!/bin/bash
set -euo pipefail

# Input CSV to evaluate (columns: prompt_id, query, domain, response)
ROOT_PATH="/data/milanlp/ceron/info-seeking/generated_responses/generated_answers_for_queries/"

model="Qwen/Qwen3.6-27B"

# Define evaluator models to run
declare -a FILES=(
  "responses_no_search/claude-sonnet-4-6_2026-04-20_15-21-52.csv"
  "responses_no_search/gpt-5.4_2026-04-21_09-48-51.csv"
  "responses_no_search/gemini-3.1-flash-lite-preview_2026-04-20_17-05-32.csv"
)

# Create directories
mkdir -p jobs logs logs_err

for file in "${FILES[@]}"; do
  safe_file_name=$(echo "$file" | sed 's|/|-|g')
  jobname="run_safety_${safe_file_name%.*}"
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
#SBATCH --time=12:00:00
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
  --input "$ROOT_PATH/$file" \
  --backend hf \
  --model "$model" \
  --local-path "/data/milanlp/huggingface/hub"

echo "--- Job Finished ---"
echo "End time: \$(date)"
EOF

  sbatch "$jobfile"
  echo "Submitted: $jobfile"
done
