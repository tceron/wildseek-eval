import glob
import os
import time
import anthropic
import pandas as pd
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils import get_experiment_start_date
from anthropic import RateLimitError

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Set ANTHROPIC_API_KEY before running this script.")
client = anthropic.Anthropic(api_key=api_key)

MODEL_NAME = "claude-sonnet-4-6"
MAX_TOKENS = 2048
POLL_INTERVAL_SECONDS = 15

def _message_text(message):
    return "".join(
        block.text for block in message.content if getattr(block, "type", "") == "text"
    )

def _wait_for_batch(batch_id: str):
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = getattr(batch, "processing_status", "")
        if status == "ended":
            return batch
        print(f"Batch {batch_id} status: {status}")
        time.sleep(POLL_INTERVAL_SECONDS)

def prompt_with_search(query: str = "What are the news of today in Italy?"):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        messages=[{"role": "user", "content": query}],
    )

    print(_message_text(response))

    print("\n--------------------------------Sources used:--------------------------------")
    for block in response.content:
        if getattr(block, "type", "") == "web_search_tool_result":
            for result in getattr(block, "content", []):
                if getattr(result, "type", "") == "web_search_result":
                    print(f"- {result.title}: {result.url}")
                    exit(0)


BATCH_SIZE = 100

def _build_prompt(row):
    return {
        "custom_id": str(row.prompt_id),
        "params": {
            "model": MODEL_NAME,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "messages": [{"role": "user", "content": row.content}],
        },
    }


def _write_batch_results(batch_id: str, writer):
    for entry in client.messages.batches.results(batch_id):
        custom_id = getattr(entry, "custom_id", "")
        result = getattr(entry, "result", None)
        if not result:
            continue
        result_type = getattr(result, "type", "")
        if result_type == "succeeded":
            generated_response = _message_text(result.message)
        else:
            generated_response = f"ERROR: {result_type}"
        writer.writerow([custom_id, generated_response])


def generate_answers_queries_batch():

    dataset = "man-annotated-train-high-risk"
    df = pd.read_csv(f'data_prompts/{dataset}.csv')
    df = df[df["high_risk_label"] != "Other"]
    df = df.drop_duplicates(subset="prompt_id")

    filename = f'generated_responses/{dataset}/{MODEL_NAME}_{get_experiment_start_date()}.csv'
    os.makedirs(f"generated_responses/{dataset}/", exist_ok=True)
    print(filename)

    all_prompts = [_build_prompt(row) for row in df.itertuples(index=False)]
    chunks = [all_prompts[i:i + BATCH_SIZE] for i in range(0, len(all_prompts), BATCH_SIZE)]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "response"])

        for i, chunk in enumerate(chunks):
            batch = client.messages.batches.create(requests=chunk)
            print(f"Batch {i + 1}/{len(chunks)} created: {batch.id}")
            _wait_for_batch(batch.id)
            _write_batch_results(batch.id, writer)
            f.flush()
            print(f"Batch {i + 1}/{len(chunks)} written.")

def _search_single(prompt_id, content, retries=3):
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=MAX_TOKENS,
                tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
                tool_choice={"type": "tool", "name": "web_search"},
                messages=[{"role": "user", "content": content}],
            )

            text = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            links = []
            for block in response.content:
                if getattr(block, "type", "") == "web_search_tool_result":
                    for result in getattr(block, "content", []):
                        if getattr(result, "type", "") == "web_search_result":
                            links.append(result.url)
            return prompt_id, text, "|||".join(links)
        
        except RateLimitError:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s
            print(f"Rate limited on {prompt_id}, retrying in {wait}s...")
            time.sleep(wait)
    return prompt_id, "ERROR: rate limit", ""


def run_claude_with_web_search(max_workers=6, max_rows=None):
    filename = f"data/responses_with_search/{MODEL_NAME}_forced_search_{get_experiment_start_date()}.csv"
    os.makedirs("data/responses_with_search", exist_ok=True)

    dataset = "man-annotated-train-high-risk"
    df = pd.read_csv(f'data_prompts/{dataset}.csv')
    df = df[df["high_risk_label"] != "Other"]
    df = df.drop_duplicates(subset="prompt_id")

    rows = list(zip(df["prompt_id"].tolist(), df["content"].tolist()))

    # Resume: skip already-processed prompt_ids from all previous output files
    done_ids = set()
    for f in glob.glob(f"data/responses_with_search/{MODEL_NAME}_forced_search_*.csv"):
        done_df = pd.read_csv(f)
        done_ids.update(done_df["prompt_id"].astype(str).tolist())

    rows = [(pid, content) for pid, content in rows if str(pid) not in done_ids]
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done, {len(rows)} remaining.")

    if max_rows is not None:
        rows = rows[:max_rows]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "response", "links"])

        futures = {}
        n_done = len(done_ids)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for prompt_id, content in rows:
                fut = executor.submit(_search_single, prompt_id, content)
                futures[fut] = prompt_id

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"Progress ({n_done}/{n_done + len(rows)} total)"):
                try:
                    prompt_id, text, links = fut.result()
                    writer.writerow([prompt_id, text, links])
                    f.flush()
                except Exception as e:
                    prompt_id = futures[fut]
                    print(f"\nERROR prompt_id={prompt_id}: {e}")

run_claude_with_web_search(max_workers=1)