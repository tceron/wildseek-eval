from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import csv
import os
import argparse


def load_model_and_tokenizer(model_name, cache_dir_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    first_device_str = next(iter(model.hf_device_map.values()))
    input_device = torch.device(first_device_str)
    return model, tokenizer, input_device


def main(model_name, model, tokenizer, input_device, batch_size=4):
    df = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    df = df[df["high_risk_label"]!="Other"]
    print(df)

    safe_model_name = model_name.replace("/", "-")
    out_dir = "generated_responses/hf_models"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{safe_model_name}.csv")

    # Resume from where we left off if the file already exists
    done_ids = set()
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        done_ids = set(existing["prompt_id"].tolist())
        print(f"Resuming: {len(done_ids)} prompts already processed.")
    else:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["prompt_id", "response"])

    df = df[~df["prompt_id"].isin(done_ids)].reset_index(drop=True)
    if df.empty:
        print("All prompts already processed.")
        return

    ids = df["prompt_id"].tolist()
    texts = df["content"].tolist()

    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            batch_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in batch_texts
            ]

            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                remove_invalid_values=True,
                renormalize_logits=True,
            )

            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

            with open(filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for pid, seq, plen in zip(batch_ids, outputs, prompt_lens):
                    gen_tokens = seq[plen:]
                    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    writer.writerow([pid, answer])

    print(f"Done. Responses saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prompt open HuggingFace models with high-risk queries.")
    parser.add_argument("-m", "--model", type=str, required=True, help="HuggingFace model name.")
    parser.add_argument("-lp", "--local_path", type=str, default="/data/milanlp/huggingface/hub",
                        help="Local cache directory for HuggingFace models.")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size for inference.")
    args = parser.parse_args()

    print(f"Loading model {args.model} from {args.local_path}...")
    model, tokenizer, input_device = load_model_and_tokenizer(args.model, args.local_path)
    main(args.model, model, tokenizer, input_device, batch_size=args.batch_size)
