# Factuality

This folder evaluates the factual accuracy of model responses using [Loki](https://github.com/Libr-AI/OpenFactVerification) (the `factcheck` package), for three models (GPT-5.4, Gemini, Claude) each run with and without web search.

## Pipeline

1. **`run_factuality.py`** — runs Loki fact-checking over a CSV of model responses and writes one JSON result per row to a JSONL file.

   ```bash
   python run_factuality.py <input.csv> <output.jsonl> [--start-row N]
   ```

   - `input.csv` must have `prompt_id` and `response` columns.
   - `--start-row` resumes processing from a given row (e.g. after a crash), appending to the existing output file.
   - Each output line contains the `prompt_id` plus Loki's fact-check result (decomposed claims, verification detail, and a `summary.factuality` score in `[0, 1]`).

2. **`plot_factuality.py`** — reads the six JSONL files in `data/`, extracts `summary.factuality` scores, and produces:
   - `results/factuality_boxplot.pdf` — boxplot of factuality scores per model/setup, with per-group means.
   - `results/factuality_summary.txt` — table of N, mean, median, std, min, max per setup.

   ```bash
   python plot_factuality.py
   ```

3. **`statistical_test.py`** — runs a one-tailed Mann-Whitney U test per model (H1: search improves factuality over base), and writes results to `results/mannwhitney_results.txt`.

   ```bash
   python statistical_test.py
   ```

## Data

`data/` contains the Loki fact-checking output already computed for the three models, with and without search:

- `gpt-5.4.jsonl` / `gpt-5.4_search.jsonl`
- `gemini.jsonl` / `gemini_search.jsonl`
- `claude.jsonl` / `claude_search.jsonl`

These files are too large for this repo, so they are hosted on OSF instead: [OSF link]. Download them into `factuality/data/` before running `plot_factuality.py` or `statistical_test.py`.

## Results

`results/` contains the outputs derived from `data/`:

- `factuality_boxplot.pdf` — boxplot comparing factuality scores across models and search settings.
- `factuality_summary.txt` — descriptive statistics per setup.
- `mannwhitney_results.txt` — Mann-Whitney U test results (does search improve factuality?) per model.
