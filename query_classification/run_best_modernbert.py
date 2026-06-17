# pip install -U transformers datasets accelerate evaluate scikit-learn
import argparse
import os

import pandas as pd
from transformers import AutoTokenizer

from common import get_task, load_model, load_query_dataframe, predict_batch, task_choices


def main():
    parser = argparse.ArgumentParser(description="Run the best ModernBERT checkpoint on all queries for a chosen task.")
    parser.add_argument("--classification-type", choices=task_choices(), required=True)
    parser.add_argument("--model-path", type=str, help="Override the task default best checkpoint path.")
    parser.add_argument("-d", "--data-path", type=str, help="Override the task default data path.")
    parser.add_argument("-f", "--file-path", type=str, help="Override the task default file stem.")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, help="Override the task default prediction output directory.")
    args = parser.parse_args()

    task = get_task(args.classification_type)
    model_path = args.model_path or task.run_default_model_path
    output_dir = args.output_dir or task.run_output_dir

    df = load_query_dataframe(task, data_path=args.data_path, file_path=args.file_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = load_model(model_path, len(task.label2id))

    pred_ids = predict_batch(
        tokenizer,
        model,
        df[task.text_column].tolist(),
        max_len=args.max_len,
        batch_size=args.batch_size,
    )
    predictions = [task.id2label[pred_id] for pred_id in pred_ids]

    results = pd.DataFrame({
        "prompt_id": df["prompt_id"].tolist() if "prompt_id" in df.columns else list(range(len(df))),
        "prediction": predictions,
    })

    os.makedirs(output_dir, exist_ok=True)
    file_stem = args.file_path or task.run_default_file_path
    results_path = os.path.join(output_dir, f"preds_{task.name}_{file_stem}.csv")
    results.to_csv(results_path, index=False)
    print(f"Saved predictions to {results_path}")


if __name__ == "__main__":
    main()
