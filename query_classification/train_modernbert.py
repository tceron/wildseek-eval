# pip install -U transformers datasets accelerate evaluate scikit-learn
import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common import disable_compilation_artifacts, get_task, load_labeled_dataframe, task_choices


class TokenizedTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_to_id, max_len):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=False,
            max_length=max_len,
        )
        self.labels = [label_to_id[label] for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def preprocess(batch, tokenizer, text_column, label_column, label_to_id, max_len):
    texts = ["" if pd.isna(value) else str(value) for value in batch[text_column]]
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_len,
    )
    encoded["labels"] = [label_to_id[label] for label in batch[label_column]]
    return encoded


def load_model_tokenizer(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        _attn_implementation="eager",
        reference_compile=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer, model


def train_one_fold(task, model_name, df_train, df_validation, output_root, max_len, seed):
    tokenizer, model = load_model_tokenizer(model_name, len(task.label_order))
    disable_compilation_artifacts(model)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    train_ds = TokenizedTextDataset(
        df_train[task.text_column].tolist(),
        df_train[task.label_column].tolist(),
        tokenizer,
        task.label2id,
        max_len,
    )
    validation_ds = TokenizedTextDataset(
        df_validation[task.text_column].tolist(),
        df_validation[task.label_column].tolist(),
        tokenizer,
        task.label2id,
        max_len,
    )

    training_args = TrainingArguments(
        output_dir=output_root,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    best_model_dir = os.path.join(output_root, "best-model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    for folder_name in os.listdir(output_root):
        folder_path = os.path.join(output_root, folder_name)
        if folder_name != "best-model" and os.path.isdir(folder_path):
            import shutil

            shutil.rmtree(folder_path)

    return metrics


def resolve_model_name(model_size: str) -> str:
    if model_size == "large":
        return "answerdotai/ModernBERT-large"
    return "answerdotai/ModernBERT-base"


def main():
    parser = argparse.ArgumentParser(description="Train ModernBERT for one of the query classification tasks.")
    parser.add_argument("--classification-type", choices=task_choices(), required=True)
    parser.add_argument("-p", "--prompts-path", type=str, help="Override the task training CSV.")
    parser.add_argument("-s", "--path-save-model", type=str, help="Override the task output root.")
    parser.add_argument("--model-sizes", type=str, help="Comma-separated sizes, for example base,large.")
    parser.add_argument("--max-lens", type=str, help="Comma-separated max lengths, for example 256,384,512.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = get_task(args.classification_type)
    output_root = args.path_save_model or task.train_save_dir
    model_sizes = [item.strip() for item in args.model_sizes.split(",")] if args.model_sizes else list(task.train_pretrained_model_names)
    max_lens = [int(item.strip()) for item in args.max_lens.split(",")] if args.max_lens else list(task.train_max_lens)

    if args.prompts_path:
        df = pd.read_csv(args.prompts_path)
        if task.text_column not in df.columns or task.source_label_column not in df.columns:
            raise KeyError(f"The provided training CSV does not match task {task.name!r}")
        df = df.dropna(subset=[task.text_column, task.source_label_column]).reset_index(drop=True)
        df[task.text_column] = df[task.text_column].astype(str)
        if task.label_map:
            df[task.label_column] = df[task.source_label_column].map(task.label_map)
        else:
            df[task.label_column] = df[task.source_label_column]
        df = df.dropna(subset=[task.label_column]).reset_index(drop=True)
    else:
        df = load_labeled_dataframe(task, mode="train")

    df_train, _ = train_test_split(
        df,
        test_size=task.default_test_size,
        stratify=df[task.label_column],
        random_state=args.seed,
    )

    os.makedirs(output_root, exist_ok=True)
    summary_rows = []

    for model_name in model_sizes:
        resolved_model_name = resolve_model_name(model_name)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        for fold_idx, (train_idx, validation_idx) in enumerate(skf.split(df_train, df_train[task.label_column]), start=1):
            df_fold_train = df_train.iloc[train_idx].copy()
            df_fold_validation = df_train.iloc[validation_idx].copy()

            for max_len in max_lens:
                run_name = f"{task.name}-{model_name}-fold{fold_idx}-len{max_len}"
                run_output_root = os.path.join(output_root, run_name)
                os.makedirs(run_output_root, exist_ok=True)

                print(f"Training {run_name}")
                metrics = train_one_fold(
                    task,
                    resolved_model_name,
                    df_fold_train,
                    df_fold_validation,
                    run_output_root,
                    max_len,
                    args.seed,
                )

                summary_rows.append(
                    {
                        "task": task.name,
                        "model_size": model_name,
                        "fold": fold_idx,
                        "max_len": max_len,
                        "eval_accuracy": metrics.get("eval_accuracy"),
                        "eval_f1": metrics.get("eval_f1"),
                    }
                )

    summary_path = os.path.join("modernbert-output", f"{task.name}_training_summary.csv")
    os.makedirs("modernbert-output", exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()