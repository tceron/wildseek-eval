"""
Factuality Boxplot Generator
----------------------------
Reads 6 JSONL files (3 models × with/without search) from factuality/data/,
extracts factuality scores, produces a styled boxplot saved to factuality/results/,
and prints a summary table with mean and median per setup.

Expected input files (in data/outputs/):
    gpt-5.4.jsonl           gpt-5.4_search.jsonl
    gemini.jsonl            gemini_search.jsonl
    claude.jsonl            claude_search.jsonl
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR  = Path("factuality/data")
PLOTS_DIR = Path("factuality/results")

SETUPS = [
    # (label,              filename,                  model_key)
    ("GPT-5.4",           "gpt-5.4.jsonl",           "gpt"),
    ("GPT-5.4\n+ Search", "gpt-5.4_search.jsonl",    "gpt"),
    ("Gemini",            "gemini.jsonl",             "gemini"),
    ("Gemini\n+ Search",  "gemini_search.jsonl",      "gemini"),
    ("Claude",            "claude.jsonl",             "claude"),
    ("Claude\n+ Search",  "claude_search.jsonl",      "claude"),
]

# Palette: one colour per model, base + search share the same hue
PALETTE = {
    "gpt":    ("#4A90D9", "#A8D0F0"),
    "gemini": ("#E8735A", "#F5B8AC"),
    "claude": ("#6BBF8E", "#B3E0C6"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scores(filepath: Path) -> tuple[list[float], list[dict]]:
    """Return (scores, missing_summary_instances) from a JSONL file."""
    scores: list[float] = []
    missing: list[dict] = []

    if not filepath.exists():
        print(f"  [WARN] File not found: {filepath}", file=sys.stderr)
        return scores, missing

    with filepath.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                instance = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {filepath.name}:{lineno} – JSON error: {exc}",
                      file=sys.stderr)
                continue

            if "summary" not in instance:
                missing.append(instance)
                continue

            try:
                score = instance["summary"]["factuality"]
                scores.append(float(score))
            except (KeyError, TypeError, ValueError) as exc:
                print(f"  [WARN] {filepath.name}:{lineno} – score error: {exc}",
                      file=sys.stderr)

    return scores, missing


def stats(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"), "std": float("nan")}
    a = np.array(scores)
    return {
        "n":      len(a),
        "mean":   float(np.mean(a)),
        "median": float(np.median(a)),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "std":    float(np.std(a)),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    all_scores:  list[list[float]] = []
    all_stats:   list[dict]        = []
    all_labels:  list[str]         = []
    all_colors:  list[str]         = []
    total_missing = 0

    print("\n=== Loading files ===")
    for i, (label, fname, model_key) in enumerate(SETUPS):
        filepath = DATA_DIR / fname
        scores, missing = load_scores(filepath)
        total_missing += len(missing)

        is_search = "_search" in fname
        color = PALETTE[model_key][1 if is_search else 0]

        all_scores.append(scores)
        all_stats.append(stats(scores))
        all_labels.append(label)
        all_colors.append(color)

        n_ok = len(scores)
        n_mis = len(missing)
        print(f"  {fname:<30}  {n_ok:>4} scores  |  {n_mis} missing summary")

    if total_missing:
        print(f"\n  ⚠  Total rows skipped (no 'summary'): {total_missing}")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n=== Summary of Results ===")
    header = f"{'Setup':<22} {'N':>5} {'Mean':>8} {'Median':>8} {'Std':>7} {'Min':>7} {'Max':>7}"
    print(header)
    print("-" * len(header))

    summary_lines = [header, "-" * len(header)]
    for label, s in zip(all_labels, all_stats):
        clean_label = label.replace("\n", " ")
        row = (f"{clean_label:<22} {s['n']:>5} {s['mean']:>8.4f} "
               f"{s['median']:>8.4f} {s['std']:>7.4f} "
               f"{s['min']:>7.4f} {s['max']:>7.4f}")
        print(row)
        summary_lines.append(row)

    # Save summary to text file
    summary_path = PLOTS_DIR / "factuality_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"\n  Summary saved → {summary_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    positions = list(range(1, len(SETUPS) + 1))

    # Subtle vertical band per model pair
    model_band_colors = ["#E8EDF5", "#FBF0EE", "#ECF7F1"]
    for i, band_color in enumerate(model_band_colors):
        x0 = 0.5 + i * 2
        ax.axvspan(x0, x0 + 2, color=band_color, alpha=0.55, zorder=0)

    bp = ax.boxplot(
        all_scores,
        positions=positions,
        patch_artist=True,
        widths=0.4,
        medianprops=dict(color="#1A1A2E", linewidth=2.2),
        whiskerprops=dict(color="#555", linewidth=1.2, linestyle="--"),
        capprops=dict(color="#555", linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.45,
                        markeredgewidth=0.5),
        boxprops=dict(linewidth=1.4),
        zorder=2,
    )

    for patch, color, flier in zip(bp["boxes"], all_colors, bp["fliers"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
        flier.set(markerfacecolor=color, markeredgecolor="#777")

    # Mean markers
    for pos, s, color in zip(positions, all_stats, all_colors):
        if s["n"] > 0:
            ax.plot(pos, s["mean"], marker="D", markersize=6,
                    color="#1A1A2E", markeredgecolor="white",
                    markeredgewidth=0.8, zorder=5, label="_nolegend_")

    # Axes styling
    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontsize=14, ha="center")
    ax.set_ylabel("Factuality Score", fontsize=14, labelpad=10)
    # ax.set_title("Factuality Scores by Model & Setup",
    #              fontsize=15, fontweight="bold", pad=16)
    ax.tick_params(axis='y', labelsize=14)

    ax.yaxis.grid(True, linestyle=":", color="#CCCCCC", alpha=0.8, zorder=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    # Model legend
    legend_handles = [
        mpatches.Patch(facecolor=PALETTE["gpt"][0],    label="GPT-5.4"),
        mpatches.Patch(facecolor=PALETTE["gemini"][0], label="Gemini"),
        mpatches.Patch(facecolor=PALETTE["claude"][0], label="Claude"),
    ]
    mean_handle = plt.Line2D([0], [0], marker="D", color="w",
                             markerfacecolor="#1A1A2E", markersize=7,
                             label="Mean")
    legend_handles.append(mean_handle)

    ax.legend(handles=legend_handles, loc="lower right",
              framealpha=0.9, fontsize=10, edgecolor="#CCCCCC")

    # Separator lines between model pairs
    for x in [2.5, 4.5]:
        ax.axvline(x, color="#BBBBBB", linewidth=1, linestyle="-", zorder=1)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "factuality_boxplot.pdf"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved    → {plot_path}\n")
    plt.show()


if __name__ == "__main__":
    main()