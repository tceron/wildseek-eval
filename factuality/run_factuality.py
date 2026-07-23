import argparse
import csv
import json
import os
from factcheck import FactCheck

def parse_args():
    parser = argparse.ArgumentParser(description="Run fact-checking on a CSV of responses.")
    parser.add_argument("input_path", help="Path to the input CSV file")
    parser.add_argument("output_path", help="Path to the output JSONL file")
    parser.add_argument("--start-row", type=int, default=0,
                         help="0-indexed data row to start processing from (skips earlier rows)")
    return parser.parse_args()

def main():
    args = parse_args()

    factcheck_instance = FactCheck(api_config={"temperature": 0})

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.input_path, "r", encoding="utf-8") as infile, \
         open(args.output_path, "a", encoding="utf-8") as outfile:
        reader = csv.DictReader(infile)

        for i, row in enumerate(reader):
            if i < args.start_row:
                continue
            prompt_id = row["prompt_id"]
            response = row["response"]
            print(f"\n=== Processing prompt_id: {prompt_id} ({i+1}) ===")

            try:
                results = factcheck_instance.check_text(response)
            except Exception as e:
                print(f"Error on prompt_id {prompt_id}: {e}")
                results = {"error": str(e)}

            output = {
                "prompt_id": prompt_id,
                **results
            }
            outfile.write(json.dumps(output) + "\n")
            outfile.flush()
            print(f"Saved result for prompt_id: {prompt_id}")

    print("\nDone! Results saved to", args.output_path)

if __name__ == "__main__":
    main()