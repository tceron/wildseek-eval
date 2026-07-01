import os
import glob
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import pandas as pd
import tqdm
import csv
from utils import get_experiment_start_date
import urllib.request

api = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api:
    raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) before running this script.")
client = genai.Client(api_key=api)

MODEL_NAME= "gemini-3.1-flash-lite-preview" #"gemini-3.1-pro-preview"
MAX_OUTPUT_TOKENS = 2048

def prompt_with_search():
    # Dynamic Grounding Configuration
    grounding_tool = types.Tool(
        google_search=types.GoogleSearchRetrieval(
            dynamic_retrieval_config=types.DynamicRetrievalConfig(
                # "AUTO": default threshold (e.g. 0.3)
                # "FORCE": sets the threshold to 0 to almost always force the search
                dynamic_threshold=0
            )
        )
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents="What are the news of today in Italy?",
        config=config,
    )
    # print(response)
    print(response.text)

    # After getting the response
    if response.candidates[0].grounding_metadata:
        metadata = response.candidates[0].grounding_metadata
        
        # 1. Get all unique links from the sources (chunks)
        if metadata.grounding_chunks:
            print("\n--------------------------------Sources used:--------------------------------")
            for chunk in metadata.grounding_chunks:
                if chunk.web:
                    print(f"- {chunk.web.title}: {chunk.web.uri}")


def run_parallel(items, worker_fn, max_workers=8, desc="Processing"):
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(worker_fn, item): idx for idx, item in enumerate(items)}
        for future in tqdm.tqdm(as_completed(future_to_idx), total=len(items), desc=desc):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results

def prompt_model_with_search(prompt: str, forced: bool = True, max_retries: int = 5):
    if forced:
        # GoogleSearch forces grounding and reliably populates grounding_chunks
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
    else:
        grounding_tool = types.Tool(
            google_search=types.GoogleSearchRetrieval(
                dynamic_retrieval_config=types.DynamicRetrievalConfig(
                    dynamic_threshold=0.3
                )
            )
        )
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )
            break
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            retryable = ("503" in str(e)) or ("429" in str(e))
            if retryable and attempt < max_retries - 1:
                wait = 2 ** attempt * 10 + random.uniform(0, 5)
                print(f"API error ({e}), retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    text = response.text or ""
    links = []
    candidate = response.candidates[0] if response.candidates else None
    if candidate and candidate.grounding_metadata:
        metadata = candidate.grounding_metadata
        # grounding_chunks holds the cited web sources; uri may be None on some chunks
        for chunk in (metadata.grounding_chunks or []):
            if chunk.web and chunk.web.uri:
                links.append(chunk.web.uri)
    return text, links

def run_gemini_with_web_search(forced: bool = True, max_workers=6, max_rows=None):
    if forced:
        filename = f"/data1/shared_datasets/project-info-seeking/generated_answers_for_queries/with_forced_search/gemini_forced_search_{get_experiment_start_date()}.csv"
    else:
        filename = f"/data1/shared_datasets/project-info-seeking/generated_answers_for_queries/with_auto_search/gemini_auto_search_{get_experiment_start_date()}.csv"

    dataset = "man-annotated-train-high-risk"
    df = pd.read_csv(f'data_prompts/{dataset}.csv')
    df = df[df["high_risk_label"] != "Other"]

    rows = list(zip(df["prompt_id"].tolist(), df["content"].tolist()))
    done_files = glob.glob("/data1/shared_datasets/project-info-seeking/generated_answers_for_queries/with_forced_search/gemini_forced_search_*.csv")
    done_ids = set()
    for f in done_files:
        try:
            done_ids.update(pd.read_csv(f)["prompt_id"].astype(str).tolist())
        except Exception:
            pass
    print(f"Found {len(done_ids)} done prompt_ids out of {len(rows)} total rows across {len(done_files)} files.")

    rows = [(pid, content) for pid, content in rows if str(pid) not in done_ids]
    if max_rows is not None:
        rows = rows[:max_rows]

    def _run_search(row):
        prompt_id, prompt = row
        text, links = prompt_model_with_search(prompt, forced=forced)
        return prompt_id, text, "|||".join(links)

    all_outputs = []
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "response", "links"])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_search, row): row for row in rows}
            for future in tqdm.tqdm(as_completed(futures), total=len(rows), desc="Processing"):
                result = future.result()
                writer.writerow(result)
                f.flush()
                all_outputs.append(result)

    return all_outputs

# outputs = run_gemini_with_web_search(forced=True, max_workers=6, max_rows=None)


def generate_answers_queries(max_workers=6, max_rows=None):

    dataset = "wildseek"
    df = pd.read_csv(f'data_prompts/{dataset}.csv')
    df = df[df["high_risk_label"] != "Other"]

    run_id = get_experiment_start_date()
    base_dir = "generated_responses"
    filename = f"{base_dir}/{MODEL_NAME}_{run_id}.csv"
    os.makedirs(base_dir, exist_ok=True)
    print(filename)

    rows = list(zip(df["prompt_id"].tolist(), df["content"].tolist()))
    if max_rows is not None:
        rows = rows[:max_rows]

    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    def _run_single(row):
        prompt_id, content = row
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=content,
                    config=config,
                )
                return prompt_id, content, response.text or ""
            except (genai_errors.ServerError, genai_errors.ClientError) as e:
                retryable = ("503" in str(e)) or ("429" in str(e))
                if retryable and attempt < 4:
                    wait = 2 ** attempt * 10 + random.uniform(0, 5)
                    print(f"API error ({e}), retrying in {wait:.1f}s (attempt {attempt + 1}/5)...")
                    time.sleep(wait)
                else:
                    raise

    success_count = 0
    error_count = 0
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "response"])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_single, row): row for row in rows}
            for future in tqdm.tqdm(as_completed(futures), total=len(rows), desc="Processing"):
                prompt_id, content = futures[future]
                try:
                    pid, query, generated_response = future.result()
                except Exception as e:
                    pid, query = prompt_id, content
                    generated_response = f"ERROR: {e}"

                # print(f"\n[Query {pid}] {query}")
                # print(f"[Response {pid}] {generated_response}")
                writer.writerow([pid, generated_response])
                f.flush()

                if str(generated_response).startswith("ERROR:"):
                    error_count += 1
                else:
                    success_count += 1

    print(f"Saved {len(rows)} rows to {filename} ({success_count} success, {error_count} errors)")

# generate_answers_queries()

def resolve_redirect(url: str) -> str:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(url)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            return e.headers.get("Location", url)
    return url




