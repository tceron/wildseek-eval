from transformers import AutoModelForSequenceClassification, AutoTokenizer

local_dir = "/data1/ceron/open-endedness-classifier"
repo_id = "tceron/open-endedness-classifier"
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = AutoModelForSequenceClassification.from_pretrained(local_dir)
tokenizer.push_to_hub(repo_id, safe_serialization=True)
model.push_to_hub(repo_id, safe_serialization=True)
