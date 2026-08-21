# High-Risk / Info-Seek / Openness Classifier Inference

Runs a Hugging Face sequence-classification model (from the Hub or a local folder) over a CSV of text and writes predictions to a CSV.

## Requirements

- Python 3.9+
- A GPU is used automatically if available (falls back to CPU otherwise)

Install dependencies:

```bash
pip install pandas torch tqdm transformers scikit-learn
```

## Data setup

The script currently reads its input from a **fixed path**, not from the `--input_csv` argument:

```
data_prompts/wildseek.csv
```

Make sure this file exists relative to where you run the script, and that it contains at least these columns:

- `content` — the text to classify
- `annotation` — the ground-truth label (rows with missing values here or in `content` are dropped)
- `prompt_id` — an identifier carried through to the output

The script splits this file 80/20 (stratified on `annotation`, seed `42`) and only runs inference on the 20% test split.

> Note: `--input_csv` is accepted on the command line but is not currently wired up to anything — pass it if you like, but it won't change what data gets loaded. If you need to point at a different file, edit the `pd.read_csv(...)` line in `main()`.

## Usage

```bash
python run_hf_classifier.py --input_csv data_prompts/wildseek.csv
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--input_csv` | *(required, currently unused — see above)* | Path to a CSV, must contain a `content` column |
| `--output_csv` | `./modernbert-output/predictions/preds_hf_high_risk_classifier.csv` | Where predictions are saved |
| `--repo_id` | `tceron/high-risk-classifier` | Hugging Face Hub repo id, or a local directory containing model/tokenizer files |

### Example: run a different model

```bash
python run_hf_classifier.py \
  --input_csv data_prompts/wildseek.csv \
  --repo_id tceron/info-seek-classifier \
  --output_csv outputs/info_seek_preds.csv
```

## Supported models / label maps

The script recognizes three specific models and maps prediction indices to label names accordingly:

- **`tceron/high-risk-classifier`** — Other, Economic and Financial, Health, Politics, Judicial and Legal, Moral Values and Religion, Security
- **`tceron/info-seek-classifier`** — information seeking, content creation, coding, not english, no request
- **`tceron/open-endedness-classifier`** — Analytical, Factual

Any other `--repo_id` (or local folder name) will raise a `ValueError`.

## Output

A CSV with:

- `id` — from `prompt_id`
- `prediction_id` — the raw predicted class index
- `prediction` — the human-readable label
