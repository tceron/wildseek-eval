"""Export the data the GitHub Pages site needs into docs/data/ (*.json and responses/*.json).

Run from the repo root after downloading the OSF data (see README):
    python docs/build_data.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data"
OUT.mkdir(exist_ok=True)

# Setups shown on the site, in display order. Files are the LLM-as-a-judge runs
# whose failure rates match Table 22 of the paper.
SETUPS = [
    ("gpt", "GPT-5.4", False, "gpt-5.4_2026-04-21_09-48-51_eval_20260512_214818.csv"),
    ("gpt_s", "GPT-5.4", True, "gpt-5.4_forced_search_tool_2026-04-28_10-58-09_eval_20260513_185522.csv"),
    ("gemini", "Gemini-3.1", False, "gemini-3.1-flash-lite-preview_2026-06-17_15-28-13_eval_20260617_160548.csv"),
    ("gemini_s", "Gemini-3.1", True, "gemini_forced_search_all_eval_20260515_111648.csv"),
    ("claude", "Claude-4.6", False, "claude-sonnet-4-6_2026-04-20_15-21-52_eval_20260512_214232.csv"),
    ("claude_s", "Claude-4.6", True, "claude-sonnet-4-6_forced_search_all_eval_20260513_183859.csv"),
    ("llama", "Llama-3.3-70B", False, "meta-llama-Llama-3.3-70B-Instruct_eval_20260515_114301.csv"),
]
CRITERIA = ["us_bias", "sycophancy", "overreliance", "vulnerable_population", "anthropomorphism", "dual_use"]
SOURCES = {"ses": "SES", "WildChat": "WildChat", "lmsys-chat-1m": "LMSYS-Chat-1M", "ShareGPT": "ShareGPT"}
MAX_INTENT_CHARS = 1500
# Responses with at least one failed criterion, split into shards so the site
# fetches only the shard of the query a reader opens. Must match app.js.
RESPONSE_SHARDS = 64


def source_of(prompt_id):
    return SOURCES.get(str(prompt_id).rsplit("_", 1)[-1], "Other")


def shard_of(prompt_id):
    return int(str(prompt_id).split("_", 1)[0]) % RESPONSE_SHARDS


def verdict_string(row):
    # One char per criterion: 1 = pass, 0 = fail, - = missing
    return "".join("-" if pd.isna(row[c]) else str(int(row[c])) for c in CRITERIA)


# Keep the paper's 3,077 rows (Table 2): drop the 5 queries without an
# open-endedness label but keep the 3 duplicated prompts, flagged with "dup"
# so per-model statistics count each prompt once.
ws = pd.read_csv(ROOT / "data_prompts" / "wildseek.csv").dropna(subset=["factual_or_analytical"])
ws["dup"] = ws.prompt_id.duplicated()
verdicts = {}
shards = [{} for _ in range(RESPONSE_SHARDS)]
for key, _, _, fname in SETUPS:
    df = pd.read_csv(ROOT / "safety_eval" / fname).drop_duplicates("prompt_id")
    verdicts[key] = dict(zip(df.prompt_id, df.apply(verdict_string, axis=1)))
    # Responses live in generated_responses/ under the eval file's run name
    failed = set(df.prompt_id[(df[CRITERIA] == 0).any(axis=1)]) & set(ws.prompt_id)
    resp = pd.read_csv(ROOT / "generated_responses" / (fname.split("_eval_")[0] + ".csv"))
    resp = resp.drop_duplicates("prompt_id").dropna(subset=["response"])
    for pid, text in zip(resp.prompt_id, resp.response):
        if pid in failed:
            shards[shard_of(pid)].setdefault(pid, {})[key] = text

queries = []
for r in ws.itertuples():
    q = {
        "id": r.prompt_id,
        "q": r.content,
        "d": r.high_risk_label,
        "t": {"Factual": "Factoid", "Analytical": "Analytical"}[r.factual_or_analytical],
        "s": source_of(r.prompt_id),
        "v": {k: verdicts[k][r.prompt_id] for k, *_ in SETUPS if r.prompt_id in verdicts[k]},
    }
    if r.dup:
        q["dup"] = 1
    queries.append(q)

intent = pd.read_csv(ROOT / "data_prompts" / "annotated-infoseek.csv")
intents = []
for r in intent.itertuples():
    text = str(r.content)
    intents.append({
        "id": r.prompt_id,
        "q": text[:MAX_INTENT_CHARS] + ("…" if len(text) > MAX_INTENT_CHARS else ""),
        "l": r.ground_truth,
        "s": source_of(r.prompt_id),
    })

meta = {
    "setups": [{"key": k, "model": m, "search": s} for k, m, s, _ in SETUPS],
    "criteria": CRITERIA,
}

for name, obj in [("queries", queries), ("intents", intents), ("meta", meta)]:
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{name}.json: {(OUT / f'{name}.json').stat().st_size / 1e3:.0f} kB")

(OUT / "responses").mkdir(exist_ok=True)
for i, shard in enumerate(shards):
    with open(OUT / "responses" / f"{i}.json", "w") as f:
        json.dump(shard, f, ensure_ascii=False, separators=(",", ":"))
print(f"responses/: {sum(len(s) for s in shards)} queries in {RESPONSE_SHARDS} shards")
