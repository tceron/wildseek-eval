from ntpath import exists
import pandas as pd
from openai import OpenAI
import glob
from high_stakes_prompts import prompts
import csv
import tqdm
import os
import sys
from utils import get_experiment_start_date
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

_openai_api_key = os.environ.get("OPENAI_API_KEY")
if not _openai_api_key:
    raise ValueError("Set OPENAI_API_KEY before running this script.")
client = OpenAI(api_key=_openai_api_key, base_url="https://eu.api.openai.com/v1")

MODEL_NAME="gpt-5.4"

def run_parallel(items, worker_fn, max_workers=8, desc="Processing"):
    """
    Run request-bound work in parallel while preserving input order.
    """
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(worker_fn, item): idx for idx, item in enumerate(items)}
        for future in tqdm.tqdm(as_completed(future_to_idx), total=len(items), desc=desc):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results

def map_response(response):
    # Keep only digits
    response = ''.join(filter(str.isdigit, response))

    mapping = {
        "1": "Politics",
        "2": "Economic and Financial",
        "3": "Security",
        "4": "Health",
        "5": "Judicial and Legal",
        "6": "Moral Values and Religion",
        "7": "Other"
    }

    label = mapping.get(response, "na")
    return label, response

def prompt_model_classification(prompt:str):
    """
    Generate a summary of a cluster of documents using OpenAI's GPT model.
    Args:
        prompt (str): The documents to summarize, formatted as a string and separated by \n.
    Returns:
        str: The summary of the documents.
    """
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
                    {"role": "developer", "content": "You are an expert in annotating text."},
                    {"role": "user", "content": prompt}], 
        max_tokens=10,
        # temperature=0.7,
    )
    response = completion.choices[0].message.content
    # print("RESPONSE:", response)
    return map_response(response) 

def prompt_model_identify_main_claim(prompt:str):
    """
    Identify the main claim from a given text using OpenAI's GPT model.
    Args:
        prompt (str): The text to analyze.
    Returns:
        str: The identified main claim.
    """
    
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
                    {"role": "developer", "content": "You are an expert in annotating text."},
                    {"role": "user", "content": prompt}], 
        max_tokens=200,
        # temperature=0.7,
    )
    return completion.choices[0].message.content


def prompt_model(prompt:str):
    """
    Generate a summary of a cluster of documents using OpenAI's GPT model.
    Args:
        prompt (str): The documents to summarize, formatted as a string and separated by \n.
    Returns:
        str: The summary of the documents.
    """
    # print(prompt)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
                    # {"role": "developer", "content": "You are an expert in annotating text."},
                    {"role": "user", "content": prompt}], 
        max_completion_tokens=2048,
        temperature=0,
        # top_p = 0.9
    )
    response = completion.choices[0].message.content or ""
    return response


def classify_high_stakes(max_workers=8):

    prompt = prompts["prompt2"]
    dataset = "high_stakes_sampled_classified"
    # Prompt
    df = pd.read_csv(f'data_prompts/pilot/{dataset}.csv')  # Assuming a CSV file with questions

    file_to_save = f"results/{dataset}-gpt4o.csv"
    if not os.path.isfile(file_to_save):
        with open(file_to_save, "w") as f:
            f.write("prompt_id,label,generated_response\n")

    texts = df.content.tolist()
    ids = df.prompt_id.tolist()

    full_prompts = [prompt.replace("<<USER_QUERY>>", text) for text in texts]
    outputs = run_parallel(
        full_prompts,
        prompt_model_classification,
        max_workers=max_workers,
        desc="Classifying prompts",
    )

    with open(file_to_save, mode='a', newline='', encoding='utf-8') as new_file:
        writer = csv.writer(new_file)
        for prompt_id, (label, generated_response) in zip(ids, outputs):
            writer.writerow([prompt_id, label, generated_response])
            

def generate_answers_queries(n_range=None, max_workers=8):

    dataset = "man-annotated-train-high-risk"
    df = pd.read_csv(f'data_prompts/{dataset}.csv')  # Assuming a CSV file with questions
    df = df[df["high_risk_label"] != "Other"]

    filename = f'generated_responses/{dataset}/{MODEL_NAME}_{get_experiment_start_date()}.csv'
    os.makedirs(f"generated_responses/{dataset}/", exist_ok=True)
    print(filename)
    # Write header once
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "response"])

    texts = df.content.tolist()
    ids = df.prompt_id.tolist()

    generated_responses = run_parallel(
        texts,
        prompt_model,
        max_workers=max_workers,
        desc="Generating responses",
    )

    with open(filename, mode='a', newline='', encoding='utf-8') as new_file:
        writer = csv.writer(new_file)
        for prompt_id, generated_response in zip(ids, generated_responses):
            writer.writerow([prompt_id, generated_response])

# for i in range(1):
#     generate_answers_queries(n_range=i, max_workers=8)

def retrieve_claims_from_queries(max_workers=8):

    df_queries = pd.read_csv(f'data_prompts/infoseek-llm.csv')  # Assuming a CSV file with questions
    df_prompts = pd.read_csv(f'claims_out/claims_2025-12-31_09-57-38.csv')
    prompt_ids = df_prompts["prompt_id"].unique().tolist()
    print(f"Total prompt ids to retrieve claims for: {len(prompt_ids)}")

    filename = f'claims_out/claims_{get_experiment_start_date()}.csv'
    print(filename)
    # Write header once
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id","run_index","model","response"])

    files = glob.glob("data/final_generated_responses/all_*_responses.csv")
    # files = ["data/final_generated_responses/all_gemma-3-27b-it_responses.csv"]
    for file in files:
        print(f"Processing file: {file}")
        df_responses = pd.read_csv(file)
        df_subset = df_responses[~df_responses["prompt_id"].isin(prompt_ids)]
        print(f"Number of matching prompt ids in this file: {len(df_subset)}")

        texts = df_subset.response.tolist()
        ids = df_subset.prompt_id.tolist()
        run_index = df_subset.run_index.tolist()
        model = file.split("all_")[1].split("_responses.csv")[0]

        prompts_to_run = [
            f"Identify the main claim in the following text:\n\n{text}\n\nProvide only the main claim."
            for text in texts
        ]
        generated_responses = run_parallel(
            prompts_to_run,
            prompt_model_identify_main_claim,
            max_workers=max_workers,
            desc=f"Extracting claims ({model})",
        )

        with open(filename, mode='a', newline='', encoding='utf-8') as new_file:
            writer = csv.writer(new_file)
            for prompt_id, idx, response in zip(ids, run_index, generated_responses):
                writer.writerow([prompt_id, idx, model, response])

# retrieve_claims_from_queries()

def prompt_model_with_web(prompt: str, forced: bool = True):

    response = client.responses.create(
        model=MODEL_NAME,
        tools=[{"type": "web_search"}],  # 👈 Enable web search
        tool_choice={"type": "web_search"} if forced else "auto", # 👈 Force using web search if forced=True, otherwise use auto
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_output_tokens=2048,
        temperature=0,
    )

    return response.output_text

def extract_links(text: str):
    pattern = r'\[.*?\]\((https?://.*?)\)'
    text = re.findall(pattern, text)
    return [link.replace("?utm_source=openai", "") for link in text]


def run_chatogpt_with_web_search(forced: bool = True, max_workers=6, max_rows=None, batch_size=50):
    if forced:
        filename = f"data/responses_with_search/{MODEL_NAME}_forced_search_tool_{get_experiment_start_date()}.csv"
    else:
        filename = f"data/responses_with_search/{MODEL_NAME}_auto_search_tool_{get_experiment_start_date()}.csv"
    os.makedirs("data/responses_with_search", exist_ok=True)

    dataset = "man-annotated-train-high-risk"
    df = pd.read_csv(f'data_prompts/{dataset}.csv')
    df = df[df["high_risk_label"] != "Other"]

    rows = list(zip(df["prompt_id"].tolist(), df["content"].tolist()))
    if max_rows is not None:
        rows = rows[:max_rows]

    def _run_web(row):
        prompt_id, prompt = row
        response = prompt_model_with_web(prompt, forced=forced)
        links = extract_links(response)
        return prompt_id, response, "|||".join(links)

    all_outputs = []
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "response", "links"])

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            batch_outputs = run_parallel(
                batch,
                _run_web,
                max_workers=max_workers,
                desc=f"Batch {i // batch_size + 1}/{(len(rows) + batch_size - 1) // batch_size}",
            )
            for prompt_id, response, links in batch_outputs:
                writer.writerow([prompt_id, response, links])
            f.flush()
            all_outputs.extend(batch_outputs)

    return all_outputs

outputs = run_chatogpt_with_web_search(forced=True, max_rows=None)
# prompt_id, response, links = outputs[0]
# print(f"prompt_id: {prompt_id}")
# print(f"response:\n{response}")
# print(f"links: {links}")
# sys.exit(0)
# # run_chatogpt_with_web_search(forced=False)
