from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# local_dir = "/data1/ceron/open-endedness-classifier"
# repo_id = "tceron/open-endedness-classifier"
# tokenizer = AutoTokenizer.from_pretrained(local_dir)
# model = AutoModelForSequenceClassification.from_pretrained(local_dir)
# tokenizer.push_to_hub(repo_id, safe_serialization=True)
# model.push_to_hub(repo_id, safe_serialization=True)

# push dataset to hub
from datasets import load_dataset

local_dir = "/home/ceron/wildseek-eval/data_prompts/annotated-infoseek.csv"
repo_id = "tceron/wildseek-5categories"
dataset = load_dataset("csv", data_files=local_dir)
dataset.push_to_hub(repo_id)
