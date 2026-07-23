"""
Mann-Whitney U Test: does search improve factuality? (one-tailed, H1: search > base)
Results saved to factuality/results/mannwhitney_results.txt
"""

import json
from pathlib import Path
from scipy import stats
import numpy as np

DATA_DIR  = Path("factuality/data")
PLOTS_DIR = Path("factuality/results")

MODELS = [
    ("GPT-5.4", "gpt-5.4.jsonl",     "gpt-5.4_search.jsonl"),
    ("Gemini",  "gemini.jsonl",       "gemini_search.jsonl"),
    ("Claude",  "claude.jsonl",       "claude_search.jsonl"),
]


def load_scores(filepath: Path) -> list[float]:
    scores = []
    with filepath.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            instance = json.loads(line)
            try:
                score = instance["summary"]["factuality"]
                if score is not None:
                    scores.append(float(score))
            except (KeyError, TypeError):
                pass
    return scores


PLOTS_DIR.mkdir(parents=True, exist_ok=True)
lines = ["Mann-Whitney U Test — H1: search > base (one-tailed)\n"]

for name, base_file, search_file in MODELS:
    base   = load_scores(DATA_DIR / base_file)
    search = load_scores(DATA_DIR / search_file)

    u_stat, p_val = stats.mannwhitneyu(search, base, alternative="greater")
    n = len(base) + len(search)
    r = stats.norm.ppf(1 - p_val) / np.sqrt(n)   # effect size

    lines.append(f"--- {name} ---")
    lines.append(f"  N base={len(base)}, N search={len(search)}")
    lines.append(f"  Mean  base={np.mean(base):.4f}, search={np.mean(search):.4f}")
    lines.append(f"  Median base={np.median(base):.4f}, search={np.median(search):.4f}")
    lines.append(f"  U={u_stat:.1f}, p={p_val:.4f}, effect r={r:.4f}\n")

out_path = PLOTS_DIR / "mannwhitney_results.txt"
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved → {out_path}")