import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from task_configs import ClassificationTaskConfig, TASK_CONFIGS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_task(task_name: str) -> ClassificationTaskConfig:
    return TASK_CONFIGS[task_name]


def task_choices() -> list[str]:
    return sorted(TASK_CONFIGS.keys())


def normalize_text_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda value: "" if pd.isna(value) else str(value))


def load_model(model_path: str, num_labels: int):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        _attn_implementation="eager",
        reference_compile=False,
    )
    model.eval()
    model.to(device)
    return model


def disable_compilation_artifacts(model):
    if hasattr(getattr(model, "model", None), "_supports_compilation"):
        model.model._supports_compilation = False

    layers = getattr(getattr(model, "model", None), "layers", None) or getattr(
        getattr(getattr(model, "model", None), "encoder", None),
        "layers",
        [],
    )
    for layer in layers:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "_orig_mod"):
            layer.mlp = layer.mlp._orig_mod
        if "compiled_mlp" in getattr(layer, "__dict__", {}) or "compiled_mlp" in getattr(layer, "_modules", {}):
            delattr(layer, "compiled_mlp")


def predict_batch(tokenizer, model, texts, max_len: int, batch_size: int = 32):
    predictions = []

    model.eval()
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[start:start + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding="longest",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())

    return predictions


def build_input_dataframe(task: ClassificationTaskConfig, data_path: str | None, file_path: str | None) -> pd.DataFrame:
    resolved_data_path = data_path or task.default_data_path
    resolved_file_path = file_path or task.default_file_path
    data_file = os.path.join(resolved_data_path, f"{resolved_file_path}.csv")

    df = pd.read_csv(data_file)
    if task.text_column not in df.columns:
        raise KeyError(f"Missing text column {task.text_column!r} in {data_file}")
    if task.source_label_column not in df.columns:
        raise KeyError(f"Missing label column {task.source_label_column!r} in {data_file}")

    df = df.dropna(subset=[task.text_column, task.source_label_column]).reset_index(drop=True)
    df[task.text_column] = normalize_text_series(df[task.text_column])

    if task.label_map:
        df[task.label_column] = df[task.source_label_column].map(task.label_map)
    else:
        df[task.label_column] = df[task.source_label_column]

    df = df.dropna(subset=[task.label_column]).reset_index(drop=True)
    return df


def load_labeled_dataframe(
    task: ClassificationTaskConfig,
    *,
    csv_path: str | None = None,
    data_path: str | None = None,
    file_path: str | None = None,
    mode: str = "train",
) -> pd.DataFrame:
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        if mode == "train":
            source_csv = task.train_csv
        elif mode == "eval":
            source_csv = task.eval_csv_path
        else:
            raise ValueError(f"Unsupported mode for labeled dataframe: {mode}")

        if source_csv.startswith("../") or source_csv.startswith("./"):
            df = pd.read_csv(source_csv)
        else:
            resolved_data_path = data_path or task.default_data_path
            resolved_file_path = file_path or task.default_file_path
            df = pd.read_csv(os.path.join(resolved_data_path, f"{resolved_file_path}.csv")) if mode == "eval" else pd.read_csv(source_csv)

    text_column = task.train_text_column if mode == "train" else (task.eval_text_column or task.train_text_column)
    label_column = task.train_label_column if mode == "train" else (task.eval_label_column or task.train_label_column)

    if text_column not in df.columns:
        raise KeyError(f"Missing text column {text_column!r}")
    if label_column not in df.columns:
        raise KeyError(f"Missing label column {label_column!r}")

    df = df.dropna(subset=[text_column, label_column]).reset_index(drop=True)
    df[text_column] = normalize_text_series(df[text_column])
    df[label_column] = df[label_column].map(task.label_map) if task.label_map else df[label_column]
    df = df.dropna(subset=[label_column]).reset_index(drop=True)
    return df


def load_query_dataframe(
    task: ClassificationTaskConfig,
    *,
    data_path: str | None = None,
    file_path: str | None = None,
) -> pd.DataFrame:
    resolved_data_path = data_path or task.run_default_data_path
    resolved_file_path = file_path or task.run_default_file_path
    data_file = os.path.join(resolved_data_path, f"{resolved_file_path}.csv")

    df = pd.read_csv(data_file)
    if task.text_column not in df.columns:
        raise KeyError(f"Missing text column {task.text_column!r} in {data_file}")

    df = df.dropna(subset=[task.text_column]).reset_index(drop=True)
    df[task.text_column] = normalize_text_series(df[task.text_column])
    return df
