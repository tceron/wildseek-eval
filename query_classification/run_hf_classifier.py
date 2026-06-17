import argparse
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LEN = 256
BATCH_SIZE = 32

def predict_batch(tokenizer, model, texts, batch_size=BATCH_SIZE):
    """Predict label indices for a list of texts."""
    all_preds = []
    model.eval()

    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
                padding="longest",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)

    return all_preds


def _label_map_key(repo_id):
    """Hub id or local folder path; basename matches hub-style names for label maps."""
    expanded = os.path.expanduser(repo_id)
    if os.path.isdir(expanded):
        return os.path.basename(os.path.abspath(expanded))
    return repo_id


def load_hf_model(repo_id):

    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
        _attn_implementation="eager",  # safer fallback for CPU
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def run_inference(data, tokenizer, model, ID_TO_LABEL):
    texts = data["content"].tolist()
    pred_ids = predict_batch(tokenizer, model, texts)
    predictions = [ID_TO_LABEL[i] for i in pred_ids]
    return pred_ids, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run info-seek classifier from Hugging Face Hub on a new dataset."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV. Must contain a 'content' column.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./modernbert-output/predictions/preds_hf_high_risk_classifier.csv",
        help="Path to save predictions CSV.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="tceron/high-risk-classifier",
        help="Hub repo id or local directory with config, weights, and tokenizer files.",
    )
    args = parser.parse_args()

    label_key = _label_map_key(args.repo_id)
    if label_key == "tceron/high-risk-classifier":
        ID_TO_LABEL = {
            0: "Other",
            1: "Economic and Financial",
            2: "Health",
            3: "Politics",
            4: "Judicial and Legal",
            5: "Moral Values and Religion",
            6: "Security",
        }
    elif label_key == "tceron/info-seek-classifier":
        ID_TO_LABEL = {
            0: "information seeking",
            1: "content creation",
            2: "coding",
            3: "not english",
            4: "no request",
        }
    elif label_key == "factual-analytical-classifier":
        ID_TO_LABEL = {
            0: "Analytical",
            1: "Factual",
        }
    else:
        raise ValueError(
            f"Invalid repo ID (or unrecognized local folder name): {args.repo_id!r} "
            f"(label map key: {label_key!r})"
        )

    map2label = {"Analytical": "Analytical", "Subjective": "Analytical", "Factual": "Factual", "Predictive": "Analytical", "Procedural": "Factual"}
    SEED = 42
    print(f"Loading model from: {args.repo_id}")
    tokenizer, model = load_hf_model(args.repo_id)
    print(f"Using device: {DEVICE}")
    print(f"Using MAX_LEN={MAX_LEN}")

    # data = pd.read_csv(args.input_csv)
    # if "content" not in data.columns:
    #     raise ValueError("Input CSV must include a 'content' column.")
    # if "id" not in data.columns:
    #     data["id"] = data.index.astype(str)

    # data = data[~data["content"].isna()].reset_index(drop=True)
    # data["content"] = data["content"].astype(str)

    df = pd.read_csv(f"data_prompts/infoseek-llm.csv")
    df = df.dropna(subset=["annotation", "content"])
    df.rename(columns={"annotation": "ground_truth"}, inplace=True)
    df["ground_truth"] = df["ground_truth"].map(map2label)
    print("df after mapping:", df)
    print(df["ground_truth"].value_counts())
    _, df_test = train_test_split(
                        df,
                        test_size=0.2,
                        stratify=df["ground_truth"],
                        random_state=SEED
                    )
    data = df_test

    tic = time.time()
    pred_ids, predictions = run_inference(data, tokenizer, model, ID_TO_LABEL)
    toc = time.time()

    results = pd.DataFrame(
        {
            "id": data["prompt_id"],
            "prediction_id": pred_ids,
            "prediction": predictions,
        }
    )

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results.to_csv(args.output_csv, index=False)

    print(f"Saved predictions to: {args.output_csv}")
    print(f"Processed {len(results)} rows in {toc - tic:.2f}s")


if __name__ == "__main__":
    main()
