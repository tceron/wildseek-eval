# pip install -U transformers datasets accelerate evaluate scikit-learn
import argparse
import os

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from common import disable_compilation_artifacts, get_task, load_labeled_dataframe, load_model, predict_batch, task_choices


def main():
    parser = argparse.ArgumentParser(description="Evaluate a ModernBERT checkpoint on a held-out split for a chosen task.")
    parser.add_argument("--classification-type", choices=task_choices(), required=True)
    parser.add_argument("--model-path", type=str, help="Checkpoint path to evaluate. Defaults to the task config.")
    parser.add_argument("-p", "--prompts-path", type=str, help="Override the task evaluation CSV.")
    parser.add_argument("-d", "--data-path", type=str, help="Override the task data path when evaluating query files.")
    parser.add_argument("-f", "--file-path", type=str, help="Override the task file stem when evaluating query files.")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--output-dir", type=str, help="Directory where evaluation CSVs are written.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = get_task(args.classification_type)
    model_path = args.model_path or task.eval_model_path or task.run_default_model_path
    output_dir = args.output_dir or task.eval_output_dir

    if args.prompts_path:
        df = pd.read_csv(args.prompts_path)
        if task.text_column not in df.columns or task.source_label_column not in df.columns:
            raise KeyError(f"The provided evaluation CSV does not match task {task.name!r}")
        df = df.dropna(subset=[task.text_column, task.source_label_column]).reset_index(drop=True)
        df[task.text_column] = df[task.text_column].astype(str)
        if task.label_map:
            df[task.label_column] = df[task.source_label_column].map(task.label_map)
        else:
            df[task.label_column] = df[task.source_label_column]
        df = df.dropna(subset=[task.label_column]).reset_index(drop=True)
    else:
        df = load_labeled_dataframe(task, data_path=args.data_path, file_path=args.file_path, mode="eval")

    df_train, df_test = train_test_split(
        df,
        test_size=task.default_test_size,
        stratify=df[task.label_column],
        random_state=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = load_model(model_path, len(task.label_order))
    disable_compilation_artifacts(model)

    test_texts = df_test[task.text_column].tolist()
    true_labels = df_test[task.label_column].tolist()
    pred_ids = predict_batch(tokenizer, model, test_texts, max_len=args.max_len, batch_size=32)
    predictions = [task.id2label[pred_id] for pred_id in pred_ids]

    report = classification_report(
        true_labels,
        predictions,
        labels=task.label_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    print(pd.DataFrame(report).transpose().round(3).to_markdown())

    if task.binary_positive_label is not None:
        binary_predictions = [task.binary_positive_label if prediction == task.binary_positive_label else f"non-{task.binary_positive_label}" for prediction in predictions]
        binary_truth = [task.binary_positive_label if truth == task.binary_positive_label else f"non-{task.binary_positive_label}" for truth in true_labels]
        binary_report = classification_report(binary_truth, binary_predictions, digits=4, output_dict=True, zero_division=0)
        print(pd.DataFrame(binary_report).transpose().round(3).to_markdown())

    os.makedirs(output_dir, exist_ok=True)
    results = pd.DataFrame({
        "prediction": predictions,
        "ground_truth": true_labels,
    })
    if "prompt_id" in df_test.columns:
        results.insert(0, "prompt_id", df_test["prompt_id"].tolist())

    results_path = os.path.join(output_dir, f"{task.name}_evaluation.csv")
    results.to_csv(results_path, index=False)
    print(f"Saved evaluation results to {results_path}")


if __name__ == "__main__":
    main()