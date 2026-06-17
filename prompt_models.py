from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import csv
import os
from utils import get_experiment_start_date
import argparse

def load_model_and_tokenizer(MODEL_NAME, cache_dir_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=cache_dir_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Figure out the "input device" (where embeddings live; usually cuda:0)
    # This is the device your input_ids must be on for sharded models.
    first_device_str = next(iter(model.hf_device_map.values()))
    input_device = torch.device(first_device_str)
    return model, tokenizer, input_device

def main(model, tokenizer, input_device, BATCH_SIZE=4):
    dataset = "missing_prompts_gemma-3-27b-it"
    df = pd.read_csv(f"data_prompts/annotations/{dataset}.csv")

    os.makedirs("generated_responses", exist_ok=True)
    filename = f'generated_responses/{dataset}-{MODEL_NAME.split("/")[-1]}_{get_experiment_start_date()}.csv'

    ids = df.prompt_id.tolist()
    texts = df.content.tolist()

    # write header once
    with open(filename, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["prompt_id", "response"])

    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_ids   = ids[i:i + BATCH_SIZE]

            # 1) Build chat prompts
            batch_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                for t in batch_texts
            ]

            # 2) Tokenize (keep on CPU, then move to the FIRST device)
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            )
            inputs = {k: v.to(input_device) for k, v in inputs.items()}

            # 3) Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                # top_k=50,  # optional; try commenting out first for stability
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                remove_invalid_values=True,
                renormalize_logits=True,
            )

            # 4) Decode ONLY the newly generated tokens per sample
            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

            with open(filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for pid, seq, plen in zip(batch_ids, outputs, prompt_lens):
                    gen_tokens = seq[plen:]  # strip prompt tokens
                    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    writer.writerow([pid, answer])


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run statements on offline models using Hugging Face.")
    parser.add_argument('-m', '--model', type=str, help="Model to run.", default= "meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('-lp', '--local_path', type=str, help="Path to upload model from.", default="/mount/arbeitsdaten/multiview/ceronte/hf_models")
    args = parser.parse_args()

    BATCH_SIZE = 4
    cache_dir_path = args.local_path
    MODEL_NAME = args.model
    print(f"Loading model {MODEL_NAME} from {cache_dir_path}...")
    model, tokenizer, input_device = load_model_and_tokenizer(MODEL_NAME, cache_dir_path)
    main(model, tokenizer, input_device, BATCH_SIZE=BATCH_SIZE)
