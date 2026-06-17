from collections import defaultdict
from fileinput import filename
import math
from urllib.parse import urlparse

import pandas as pd
import glob
import numpy as np
from utils import map_response2label
import re
import os
# from datasets import load_dataset
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
import os

def get_info_seeking_queries():
    files = glob.glob("results/google--gemma-3-27b-it_*.csv")
    df_all = pd.DataFrame()

    for file in files:
        try:
            # Use on_bad_lines='skip' to skip problematic rows
            df = pd.read_csv(file, on_bad_lines='skip')
            df["response_mapped"] = df["response"].apply(lambda x: map_response2label(str(x)))
            df_all = pd.concat([df_all, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # Drop duplicates
    df_all = df_all.drop_duplicates(subset=["prompt_id", "dataset"], keep="last")
    df_all = df_all[df_all["response_mapped"]=="information seeking"]
    print(df_all)

    files = glob.glob("/data1/shared_datasets/user_conversations/*_history.csv")
    df = pd.DataFrame()
    for file in files:
        try:
            # Use on_bad_lines='skip' to skip problematic rows
            df_temp = pd.read_csv(file, on_bad_lines='skip')
            df_temp["dataset"] = file.split("/")[-1].replace("_history.csv", "")
            df = pd.concat([df, df_temp], ignore_index=True)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            
    df_all = df_all.merge(df[["prompt_id", "conversation_id", "dataset", "content", "conversation_history"]], on=["prompt_id", "dataset"], how="inner")
    print(df_all.dropna(subset=["content"]))
    print(df_all.response_mapped.value_counts())
    print(df_all)
    df_all.to_csv("/data1/shared_datasets/user_conversations/info-seeking.csv", index=False)

    return df_all

def fix_col_history():
    df = pd.read_csv("data_prompts/gold_standard_test_set.csv")
    history = []
    for _, row in df.iterrows():
        if str(row["history"])!="nan":
            history.append(row["history"])
        else:
            history.append(row["conversation_history"])

    df["conversation_history"] = history
    df = df.drop(columns=["history"])
    df.to_csv("data_prompts/gold_standard_test_set.csv", index=False)

# df = pd.read_csv("data_prompts/gold_standard_test_set.csv")
# print(df.ground_truth.unique())
# df = df[df["ground_truth"]=="Info-seeking"]
# df.to_csv("data_prompts/info-seeking_test_set.csv", index=False)

def get_sample():
    df_final = pd.DataFrame()
    df = pd.read_csv("/data1/shared_datasets/user_conversations/info-seeking.csv")
    for dataset in df.dataset.unique():
        df_temp = df[df["dataset"]==dataset]
        sample = df_temp.sample(n=20, random_state=50)
        df_final = pd.concat([df_final, sample], ignore_index=True)
    df_final.to_csv("./data_prompts/sample_pred_info_seeking.csv", index=False)
    print(df_final)


def all_prompts_predicted(dataset):
    files = glob.glob("results/info_seeking_preds/google--gemma-3-27b-it_*.csv")
    df_all = pd.DataFrame()

    for file in files:
        try:
            # Use on_bad_lines='skip' to skip problematic rows
            df = pd.read_csv(file, on_bad_lines='skip')
            df["response_mapped"] = df["response"].apply(lambda x: map_response2label(str(x)))
            df_all = pd.concat([df_all, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # Drop duplicates
    df_all = df_all.drop_duplicates(subset=["prompt_id", "dataset"], keep="last")
    # print(df_all)

    files = [f"/data1/shared_datasets/user_conversations/{dataset}_history.csv"]
    # print(files)
    df = pd.DataFrame()
    for file in files:
        try:
            # Use on_bad_lines='skip' to skip problematic rows
            df_temp = pd.read_csv(file, on_bad_lines='skip')
            df_temp["dataset"] = file.split("/")[-1].replace("_history.csv", "")
            df = pd.concat([df, df_temp], ignore_index=True)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    # print(df)
     
    df_all = df_all.merge(df[["prompt_id", "conversation_id", "dataset", "content", "conversation_history"]], on=["prompt_id", "dataset"], how="inner")
    df_all = df_all[df_all["dataset"]==dataset]
    print(df_all.dropna(subset=["content"]))
    print(df_all.response.unique())
    print(df_all)
    df_all.to_csv(f"/data1/shared_datasets/user_conversations/{dataset}_predicted.csv", index=False)

    return df_all

def process_stakes_dataset():
    df = pd.read_csv("data_prompts/annotations/sample_stakes_info_seeking.csv")
    df = df[['dataset', 'prompt_id', 'response_mapped', 'conversation_id', 'content','conversation_history',"stakes-topic-gold"]]
    df["stakes-topic-gold"] = df["stakes-topic-gold"].fillna("other")
    df["stakes-topic-gold"] = df["stakes-topic-gold"].apply(lambda x: x.lower())
    df["stakes-topic-gold"] = df["stakes-topic-gold"].apply(lambda x: x.split(".")[-1].rstrip().lstrip())
    print(df["stakes-topic-gold"].value_counts())
    df.to_csv("data_prompts/stakes_testset.csv", index=False)

def get_sample_info_seeking(): 
    df = pd.read_csv("/data1/shared_datasets/user_conversations/info-seeking.csv")
    df_final = pd.DataFrame()
    for dataset in df.dataset.unique():
        if dataset=="en_elisa":
            tmp = df[df["dataset"]=="en_elisa"]
        else:
            tmp = df[df["dataset"]==dataset]
            tmp = tmp.sample(n=8000, random_state=50)
        df_final = pd.concat([df_final, tmp], ignore_index=True)
    df_final.to_csv("data_prompts/sample_info_seeking.csv", index=False)

def match_preds_queries():
    info_seeking = pd.read_csv("data_prompts/sample_info_seeking.csv").drop(columns=["response_mapped", "response"])
    preds = pd.read_csv("results/stakes/meta-llama--Llama-3.1-70B-Instruct_2025_06_15_18_40_42.csv")
    # Merge the two DataFrames on 'prompt_id' and 'dataset'
    merged_df = info_seeking.merge(preds, on=["prompt_id"], how="inner")
    print(merged_df.response_mapped.value_counts())
    # Select relevant columns
    merged_df = merged_df[['prompt_id', 'dataset', 'response_mapped', 'conversation_id', 'content', 'conversation_history']]
    merged_df = merged_df[merged_df["response_mapped"]=="high-stakes"]
    merged_df.to_csv("data_prompts/annotations/sample_info_seeking_preds_llama70b.csv", index=False)


def llm_responses():
    df = pd.read_csv("results/stakes_chat_responses_llama+oai.csv")
    print(df.stakes_gold.value_counts())
    df = df[df["stakes_gold"]=="High-stakes"]
    for _, row in df.iterrows():
        print(row["stakes_gold"], row["content"])
        print("~~~LLAMA:", row["llama_responses"], "\n\n")
        print("~~~openAI:",row["oai_responses"])
        print("===" * 50)
        input("Press Enter to continue...")


def is_link_only(s):
    # Check if the entire string is just a URL
    return re.fullmatch(r'https?://\S+', s) is not None


def has_english_word(s):
    # Checks for at least one English word (a-zA-Z)
    return re.search(r'\b[a-zA-Z]{2,}\b', s) is not None

def process_orcas():
    # Read the TSV file with tab as the delimiter
    df = pd.read_csv("data/orcas-doctrain-queries.tsv", sep="\t", header=None, names=["query_id", "query"])
    # remove rows with "query" column as NaN and length of "query" less than 3
    df = df.dropna(subset=["query"])
    dic_strings = dict(zip(df["query_id"], df["query"]))

    result = []
    for id, s in dic_strings.items():
        if not is_link_only(s) and has_english_word(s):
            result.append((id, s))
        else:
            print(f"Skipping query_id {id} with content: {s}")
    filtered_queries = pd.DataFrame(result, columns=["query_id", "query"])
    print(filtered_queries)
    filtered_queries.to_csv("/data1/shared_datasets/web_search/orcas_filtered_queries.csv", index=False)

def get_marcos():
    # Load dataset from Hugging Face
    df_final = pd.DataFrame()
    for set in ["train", "validation", "test"]:
        hf_dataset = load_dataset("microsoft/ms_marco", 'v2.1', split=set)  # Replace with the actual dataset name
        df_hf = hf_dataset.to_pandas()  # Convert Hugging Face dataset to pandas DataFrame
        df_hf = df_hf[["query_id", "query"]]
        df_hf = df_hf.dropna(subset=["query"])
        df_final = pd.concat([df_final, df_hf], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["query_id", "query"], keep="last")
    df_final.to_csv("/data1/shared_datasets/web_search/ms-marco_filtered_queries.csv")
    print(df_final)

def natural_questions_dataset():

    file_path = 'data/simplified-nq-train.jsonl'  # Replace with your actual file path
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data=json.loads(line)
            # Now `data` is a list of dictionaries (one per line)
            results.append((data["example_id"],data["question_text"]))
    print(len(results))
    df = pd.DataFrame(results, columns=["query_id", "query"])
    df = df.dropna(subset=["query"])
    df = df.drop_duplicates(subset=["query_id", "query"], keep="last")
    df.to_csv("/data1/shared_datasets/web_search/natural_questions_filtered_queries.csv", index=False)
    print(df)

def add_more_data_test_set():
    label2name = {'content creation':'Content creation', 'reformulation':'Reformulation', 'coding':'Coding', 'not English':'Not English'}
    df = pd.read_csv("data_prompts/gold_standard_test_set.csv")
    tmp = pd.read_csv("data/more_info_seeking_annotations.csv")
    tmp["ground_truth"] = tmp["comments_tanise"].apply(lambda x: label2name.get(x, x) if x in label2name else "Info-seeking")
    tmp = tmp[['prompt_id', 'dataset', 'conversation_id', 'content', 'ground_truth', 'conversation_history']]
    df_final = pd.concat([df, tmp], ignore_index=True)
    df_final.to_csv("data_prompts/gold_standard_test_set_extended.csv", index=False)
    print(df_final)
    print(df_final.ground_truth.value_counts())


def get_training_sample():
    files = glob.glob("results/info_seeking_preds/google--gemma-3-27b-it_*.csv")
    df_all = pd.DataFrame()

    for file in files:
        try:
            # Use on_bad_lines='skip' to skip problematic rows
            df = pd.read_csv(file, on_bad_lines='skip')
            df["response_mapped"] = df["response"].apply(lambda x: map_response2label(str(x)))
            df_all = pd.concat([df_all, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading file {file}: {e}")

    # Drop duplicates
    df_all = df_all.drop_duplicates(subset=["prompt_id", "dataset"], keep="last")

    files = glob.glob("/data1/shared_datasets/project-info-seeking/user_conversations/*_history.csv")
    df = pd.DataFrame()
    for file in files:
        try:
            # Use on_bad_lines='skip' to skip problematic rows
            df_temp = pd.read_csv(file, on_bad_lines='skip')
            df_temp["dataset"] = file.split("/")[-1].replace("_history.csv", "")
            df = pd.concat([df, df_temp], ignore_index=True)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
            
    df_all = df_all.merge(df[["prompt_id", "conversation_id", "dataset", "content", "conversation_history"]], on=["prompt_id", "dataset"], how="inner")
    df_sample = pd.DataFrame()
    for dataset in df_all.dataset.unique():
        temp = df_all[df_all["dataset"]==dataset]
        tmp =  temp.sample(n=1200, random_state=50)
        df_sample = pd.concat([df_sample, tmp], ignore_index=True)
    df_sample.rename(columns={"response_mapped":"ground_truth"}, inplace=True)
    df_sample = df_sample[['prompt_id', 'dataset', 'conversation_id', 'content', 'ground_truth', 'conversation_history']]
    df_sample = df_sample[df_sample["ground_truth"]!="na"].dropna(subset=["ground_truth"])
    df_sample.to_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/training_sample_{len(df_sample)}.csv", index=False)

    # return df_all

def process_validated_training_set():
    df = pd.read_csv("data_prompts/training_sample_validated.csv")
    print(df)
    # print(df.ground_truth.value_counts())
    df_test = pd.read_csv("data_prompts/gold_standard_test_set_extended.csv")
    df_test["ground_truth"] = ["Content creation" if x=="Reformulation" else x for x in df_test["ground_truth"]]
    print(df_test.ground_truth.value_counts())
    df_test.to_csv("data_prompts/gold_standard_test_set_extended_validated.csv", index=False)

def add_unique_id_col_complete_datasets():
    files = glob.glob("/data1/shared_datasets/project-info-seeking/user_conversations/*_history.csv")
    for file in files:
        df = pd.read_csv(file)
        dataset = file.split("/")[-1].replace("_history.csv", "")
        print(dataset)
        df["prompt_id"] = df.apply(lambda row: "_".join([str(int(row["prompt_id"])), dataset.replace("en_", "")]), axis=1)
        print(df)
        # df.to_csv(file, index=False)

def add_unique_id_col_training_datasets():
    files = ["data/training_sample_berat.csv"] #glob.glob("data_prompts/*set*.csv")
    # files.extend(glob.glob("data_prompts/*training*.csv"))
    for file in files:
        print(file)
        df = pd.read_csv(file)
        df["prompt_id"] = df.apply(lambda row: "_".join([str(int(row["prompt_id"])), row["dataset"].replace("en_", "")]), axis=1)
        print(df)
        # df.to_csv(file, index=False)

def deduplicate_train_test_datasets():
    df_train = pd.read_csv("data_prompts/training_sample_validated.csv")
    df_test = pd.read_csv("data_prompts/gold_standard_test_set_extended_validated.csv")

    # check if there are any common prompt_ids in df_train and df_test
    common_prompt_ids = set(df_train["prompt_id"]).intersection(set(df_test["prompt_id"]))
    # print(f"Number of common prompt_ids: {len(common_prompt_ids)}")
    df_test = df_test[~df_test["prompt_id"].isin(common_prompt_ids)]
    print(df_test)

    elisa = pd.read_csv("/data1/shared_datasets/project-info-seeking/user_conversations/en_elisa_history.csv")
    common_prompt_ids = set(pd.concat([df_train["prompt_id"], df_test["prompt_id"]])).intersection(set(elisa["prompt_id"]))
    # sample 100 from elisa that are not in common_prompt_ids
    print(df_test.columns)
    elisa_sample = elisa[~elisa["prompt_id"].isin(common_prompt_ids)].sample(n=1000-len(df_test), random_state=50)
    elisa_sample['dataset'] = ['en_elisa']*len(elisa_sample)
    elisa_sample["ground_truth"] = ["Info-seeking"]*len(elisa_sample)
    df_test = pd.concat([df_test, elisa_sample[['prompt_id', 'dataset', 'conversation_id', 'content', 'ground_truth',
       'conversation_history']]], axis=0, ignore_index=True)
    # df_test.to_csv("data_prompts/gold_standard_test_set_extended_validated.csv", index=False)

def gather_all_datasets():
    df1= pd.read_csv("data_prompts/gold_standard_test_set_extended_validated.csv")
    df2 = pd.read_csv(f"data_prompts/training_sample_validated.csv")
    df3 = pd.read_csv("data/training_sample_berat.csv")
    df = pd.concat([df1, df2, df3])
    df = df.drop_duplicates(subset=["content"])
    print(df)
    res = df.sample(frac=1).reset_index(drop=True)
    res.to_csv("./data_prompts/info-non-info-dataset.csv")

def read_dataset():
    df= pd.read_csv("data_prompts/gold_standard_test_set.csv")
    df_ground_truth = pd.read_csv("./data_prompts/info-non-info-dataset.csv", index_col=False)
    print(df.ground_truth.value_counts())
    dic_ground_truth = dict(zip(df_ground_truth["prompt_id"], df_ground_truth["ground_truth"]))
    df["ground_truth"] = df["prompt_id"].apply(lambda x: dic_ground_truth.get(x, "NA"))
    df = df[df["ground_truth"]!="NA"]
    print(df.ground_truth.value_counts())
    print(df)
    # df.to_csv("data_prompts/gold_standard_test_set.csv", index=False)

def map_queries_preds(dataset):
    preds = pd.read_csv(f"classification-infoseek/modernbert-output/predictions/preds_{dataset}_history.csv")
    print(preds)
    history = pd.read_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/{dataset}_history.csv")
    # history["prompt_id"] = history.apply(lambda row: "_".join([str(int(row["prompt_id"])), f"{dataset.split("_")[1]}"]), axis=1)
    merged_df = history.merge(preds[["prompt_id", "prediction"]], on=["prompt_id"], how="inner")
    # merged_df["prompt_id"] = merged_df.apply(lambda row: "_".join([str(int(row["prompt_id"])), f"{dataset.split("_")[1]}"]), axis=1)

    ground_truth = pd.read_csv("data_prompts/gold_standard_test_set.csv")
    ids = set(ground_truth["prompt_id"].tolist())&(set(merged_df["prompt_id"].tolist()))
    id2label = dict(zip(ground_truth["prompt_id"], ground_truth["ground_truth"]))
    print(merged_df.prediction.value_counts())
    merged_df["prediction"] = merged_df.apply(
        lambda row: id2label[row["prompt_id"]] if row["prompt_id"] in ids else row["prediction"], axis=1
    )
    print(merged_df.prediction.value_counts())
    print(merged_df)
    
    merged_df.to_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/modernbert_predicted/preds_{dataset}_mapped.csv", index=False)


def prepare_data_for_annotation():
    df = pd.read_csv("data/annotations/Knowledge_Intent_Annotations_agreement.csv")
    remove_ids = df.prompt_id.unique().tolist()
    df_annotations = pd.DataFrame()
    df_high_stakes = pd.read_csv("data/high-stakes-classification-gpt4o.csv")
    df_high_stakes = df_high_stakes[(df_high_stakes["label"]!="Other")&(df_high_stakes["label"]!="na")]
    df = pd.read_csv("data_prompts/high-stakes-queries.csv")
    df_high_stakes = df_high_stakes.merge(df[["prompt_id", "content"]], on=["prompt_id"], how="inner")

    proportion_topics = (df_high_stakes.label.value_counts()/len(df_high_stakes)).to_dict()

    df_high_stakes["dataset"] = df_high_stakes["prompt_id"].apply(lambda x: x.split("_")[-1])
    df_tmp = df_high_stakes[(~df_high_stakes["prompt_id"].isin(remove_ids))]
    for topic, proportion in proportion_topics.items():
        n = int(3000*proportion)
        df_elisa = df_tmp[(df_tmp["label"]==topic) & (df_tmp["dataset"]=="elisa")]
        if len(df_elisa)>=n:
            df_sample_elisa = df_elisa.sample(n=n, random_state=50)
        else:
            df_sample_elisa = df_elisa
            df_sample_elisa = pd.concat([df_sample_elisa, df_tmp[(df_tmp["label"]==topic) & (df_tmp["dataset"]!="elisa")].sample(n=n-len(df_elisa), random_state=50)], ignore_index=True)
        df_annotations = pd.concat([df_annotations, df_sample_elisa], ignore_index=True)

    df_annotations = df_annotations[["prompt_id", "label", "content"]]

    first_sample, remaining_df = train_test_split(
        df_annotations,
        train_size=1000,
        stratify=df_annotations["label"],
        random_state=50
    )
    first_sample.to_csv("data_prompts/annotations/info_seeking_data_for_annotation_1000.csv", index=False)

    first_annotator, remaining_df = train_test_split(
        remaining_df,
        train_size=300,
        stratify=remaining_df["label"],
        random_state=50
    )
    # first_annotator.to_csv("data_prompts/annotations/info_seeking_data_for_annotation_berat.csv", index=False)

    second_annotator, third_annotator = train_test_split(
        remaining_df,
        train_size=850,
        stratify=remaining_df["label"],
        random_state=50
    )
    print(first_sample)
    print(first_annotator)
    print(second_annotator)
    print(third_annotator)
    second_annotator.to_csv("data_prompts/annotations/info_seeking_data_for_annotation_ivana.csv", index=False)
    third_annotator.to_csv("data_prompts/annotations/info_seeking_data_for_annotation_lea.csv", index=False)

    df = pd.concat([first_sample, first_annotator, second_annotator, third_annotator], ignore_index=True)
    df.to_csv("data_prompts/annotations/info_seeking_data_for_annotation_total_3000.csv", index=False)

def calculate_cohens_kappa():
    files = glob.glob("data/annotations/query_type/annotations_round*.csv")
    for file in files: 
        tmp = pd.read_csv(file)
        kappa = cohen_kappa_score(tmp['label_annotator1'], tmp['label_annotator2'])
        print(f"Cohen's Kappa for {file}:\n{kappa}")


def join_infoseek_with_validation():
    """
    Create a DataFrame that combines `data_prompts/infoseek-llm.csv`
    with `data/validation_high_stakes.csv`, replacing any rows in the
    first file whose `prompt_id` also appears in the second file.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame where validated rows overwrite originals.
    """
    df_infoseek = pd.read_csv("data_prompts/infoseek-llm.csv")
    id2annotation = dict(zip(df_infoseek["prompt_id"], df_infoseek["annotation"]))
    print(id2annotation)
    df_validation = pd.read_csv("data/annotations/validated_annotations_high_stakes.csv")
    df_validation.rename(columns={"label_annotated":"label"}, inplace=True)

    df_validation["annotation"] = df_validation["prompt_id"].apply(lambda x: id2annotation.get(x, "NA"))
    df_validation = df_validation[df_validation["annotation"]!="NA"]
    # print set of prompt_ids in df_validation that are not in id2annotation
    print(len(set(df_validation["prompt_id"]) - set(id2annotation.keys())))

    if "prompt_id" not in df_infoseek.columns or "prompt_id" not in df_validation.columns:
        raise ValueError("Both dataframes must contain a 'prompt_id' column.")

    # Keep only rows from infoseek whose prompt_id is NOT in validation
    infoseek_filtered = df_infoseek[~df_infoseek["prompt_id"].isin(df_validation["prompt_id"])]

    # Concatenate filtered infoseek rows with validation rows (validation overwrites by construction)
    df_combined = pd.concat([infoseek_filtered, df_validation], ignore_index=True)
    # drop Unnamed: 0 column if exists      
    if "Unnamed: 0" in df_combined.columns:
        df_combined = df_combined.drop(columns=["Unnamed: 0", "agreement"])

    df_combined.to_csv("data_prompts/infoseek-llm-with-validation.csv", index=False)
    print(df_combined)

def match_content_infoseek_llm(df_llmsys, x):
    try: 
        prompt_id = df_llmsys[df_llmsys["content"]==x]["prompt_id"].values[0]
        return prompt_id
    except Exception as e:
        print(f"Error: {e}")
        print(x)
        return "NA"


def correct_prompt_ids_lmsys_chat_1m():
    df_llmsys = pd.read_csv("/data1/shared_datasets/project-info-seeking/user_conversations/en_lmsys-chat-1m_history.csv")
    df_infoseek = pd.read_csv("data_prompts/infoseek-llm.csv").drop(columns=["Unnamed: 0"])
    # create filtered df_infoseek where column "dataset" contains stirng "lmsys-chat-1m"
    df_infoseek_lmsys = df_infoseek[df_infoseek["prompt_id"].str.contains("lmsys-chat-1m")]
    full_dataset = df_infoseek[~df_infoseek["prompt_id"].str.contains("lmsys-chat-1m")]
    # Replace the prompt_id from df_infoseek_lmsys with the prompt_id from df. Match datasets with the "content" column
    df_infoseek_lmsys["prompt_id"] = df_infoseek_lmsys.apply(lambda row: match_content_infoseek_llm(df_llmsys, row["content"]), axis=1)
    # df_infoseek_lmsys = df_infoseek_lmsys[df_infoseek_lmsys["prompt_id"]!="NA"]
    full_dataset = pd.concat([full_dataset, df_infoseek_lmsys], ignore_index=True)
    # change string "elisa" string in prompt_id column to "ses"
    full_dataset["prompt_id"] = full_dataset["prompt_id"].str.replace("elisa", "ses")
    print(full_dataset)
    full_dataset.to_csv("data_prompts/infoseek-llm.csv", index=False)

def remove_duplicates_lmsys_chat_1m():
    df_llmsys = pd.read_csv("/data1/shared_datasets/project-info-seeking/user_conversations/en_lmsys-chat-1m_history.csv")
    print(df_llmsys)
    df_llmsys = df_llmsys.drop_duplicates(subset=["content"], keep="first")
    print(df_llmsys)
    df_llmsys.to_csv("/data1/shared_datasets/project-info-seeking/user_conversations/en_lmsys-chat-1m_history.csv", index=False)

def dataset_with_history():
    files = ["/data1/shared_datasets/project-info-seeking/user_conversations/datasets_filtered/en_WildChat_history.csv", "/data1/shared_datasets/project-info-seeking/user_conversations/datasets_filtered/en_ShareGPT_history.csv", "/data1/shared_datasets/project-info-seeking/user_conversations/datasets_filtered/en_SES_history.csv"]
    df_all = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        df = df[['prompt_id', 'content']]
        df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all

def gather_infoseek_llm_train_high_risk():
    df_all = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    other_df = pd.read_csv("data/high-stakes-classification-gpt4o.csv")
    # dataset = [i.split("_")[-1] for i in other_df.prompt_id.tolist()]
    other_df = other_df[other_df["label"]=="Other"].sample(n=1200, random_state=50)
    # remove the ones where prompt_id contains "lmsys-chat-1m"
    other_df = other_df[~other_df["prompt_id"].str.contains("lmsys-chat-1m")]
    other_df = other_df.merge(dataset_with_history(), on="prompt_id", how="left")
    other_df.rename(columns={"label":"high_risk_label"}, inplace=True)
    df_all.rename(columns={"label_validated":"high_risk_label"}, inplace=True)

    df_all = pd.concat([df_all[['prompt_id', 'content', 'high_risk_label']], other_df[['prompt_id', 'content', 'high_risk_label']]], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["content"], keep="last")
    df_all = df_all[~df_all["high_risk_label"].isna()]
    print(df_all)
    df_all.to_csv("data_prompts/train-high-risk.csv", index=False)
    print(df_all.high_risk_label.value_counts())

def _sample_infoseek_prioritizing_ses(df, n, random_state=50):
    """Sample up to n rows, maximizing rows whose prompt_id ends with '_ses'."""
    if len(df) <= n:
        return df
    ses_mask = df["prompt_id"].astype(str).str.endswith("_ses")
    df_ses = df[ses_mask]
    df_other = df[~ses_mask]
    k_ses = min(n, len(df_ses))
    if k_ses == 0:
        return df_other.sample(n=n, random_state=random_state).reset_index(drop=True)
    if k_ses < len(df_ses):
        part_ses = df_ses.sample(n=k_ses, random_state=random_state)
    else:
        part_ses = df_ses
    need = n - len(part_ses)
    if need == 0:
        return part_ses.reset_index(drop=True)
    part_other = df_other.sample(n=need, random_state=random_state)
    return pd.concat([part_ses, part_other], ignore_index=True)

map2label = {"Analytical": "Analytical", "Subjective": "Analytical", "Factual": "Factual", "Predictive": "Analytical", "Procedural": "Analytical"}

def join_infoseek_with_high_risk():
    final_df = pd.DataFrame()
    df_infoseek = pd.read_csv("data_prompts/infoseek-llm.csv")
    df_high_risk = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    df_infoseek = df_infoseek.merge(df_high_risk[["prompt_id", "high_risk_label"]], on="prompt_id", how="left").drop(columns=["label"])
    df_infoseek["annotation"] = df_infoseek["annotation"].map(map2label)
    df_infoseek.dropna(subset=["high_risk_label", "annotation"], inplace=True)
    for label in df_infoseek["high_risk_label"].unique():
        if label == "Other":
            continue
        for annotation in df_infoseek[df_infoseek["high_risk_label"]==label]["annotation"].unique():
            df_tmp = df_infoseek[(df_infoseek["high_risk_label"]==label) & (df_infoseek["annotation"]==annotation)]
            print(len(df_tmp), label, annotation)
            df_tmp = _sample_infoseek_prioritizing_ses(
                df_tmp, min(90, len(df_tmp)), random_state=50
            )
            final_df = pd.concat([final_df, df_tmp], ignore_index=True)
    print(len(final_df))
    # remove the rows in "content" that are shorter than 13 characters
    final_df = final_df[final_df["content"].str.len()>=13]
    # remove the rows in "content" that have "NAME_*" in the content where * is a number
    final_df = final_df[~final_df["content"].str.contains(r"NAME_\d+")]
    print(len(final_df))
    print(final_df.annotation.value_counts())
    print(final_df.high_risk_label.value_counts())
    final_df.to_csv("data_prompts/infoseek-llm-with-high-risk.csv", index=False)

def create_csv_samples_for_annotations():
    df = pd.read_csv("data/annotations/4-topics-infoseek-llm-with-high-risk.csv")
    output_dir = "data/info_seeking_behavior_task"
    os.makedirs(output_dir, exist_ok=True)

    required_columns = {"prompt_id", "content", "annotation_v2", "high_risk_label"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["annotation_v2"] = df["annotation_v2"].astype(str).str.strip()
    df = df[df["annotation_v2"].isin(["Factual", "Analytical"])].copy()
    topics = sorted(df["high_risk_label"].dropna().unique().tolist())

    if len(topics) != 3:
        raise ValueError(
            f"Expected exactly 3 topics in 'high_risk_label', found {len(topics)}: {topics}"
        )

    # 8 unique query sets, each duplicated 10 times => 80 files total.
    n_unique_sets = 8
    duplicates_per_set = 10
    global_file_idx = 1

    for set_idx in range(n_unique_sets):
        set_parts = []
        for topic_idx, topic in enumerate(topics):
            topic_factual = df[
                (df["high_risk_label"] == topic) & (df["annotation_v2"] == "Factual")
            ]
            topic_analytical = df[
                (df["high_risk_label"] == topic) & (df["annotation_v2"] == "Analytical")
            ]

            if len(topic_factual) < 2 or len(topic_analytical) < 2:
                raise ValueError(
                    f"Not enough rows for topic '{topic}': "
                    f"Factual={len(topic_factual)}, Analytical={len(topic_analytical)}"
                )

            factual_sample = topic_factual.sample(
                n=2, random_state=50 + set_idx * 100 + topic_idx * 2
            )
            analytical_sample = topic_analytical.sample(
                n=2, random_state=51 + set_idx * 100 + topic_idx * 2
            )
            set_parts.append(pd.concat([factual_sample, analytical_sample], ignore_index=True))

        set_df = pd.concat(set_parts, ignore_index=True)
        set_df = set_df.sample(frac=1, random_state=900 + set_idx).reset_index(drop=True)

        for copy_idx in range(duplicates_per_set):
            output_path = os.path.join(
                output_dir,
                f"queries_for_annotation_{global_file_idx:03d}.csv",
            )
            set_df = set_df[["prompt_id", "content"]]
            set_df["answer"] = [""]*len(set_df)
            set_df.to_csv(output_path, index=False)
            global_file_idx += 1

    print(f"Created {global_file_idx - 1} CSV files in '{output_dir}'.")

def get_factual_queries():
    df = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    ids = df[df["annotation"]=="Factual"]["prompt_id"].tolist()
    files = glob.glob("generated_responses/man-annotated-train-high-risk/*.csv")
    output_folder = "generated_responses/factual_queries"
    os.makedirs(output_folder, exist_ok=True)
    for file in files:
        filename = file.split("/")[-1]
        df_temp = pd.read_csv(file)
        df_temp = df_temp[df_temp["prompt_id"].isin(ids)]
        # df_temp.to_csv(f"{output_folder}/{filename}", index=False)
        print(df_temp)

def annotate_generated_responses():
    df = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    id2domain = dict(zip(df["prompt_id"], df["high_risk_label"]))
    id2query = dict(zip(df["prompt_id"], df["content"]))
    # print(id2domain)
    files = glob.glob("generated_responses/man-annotated-train-high-risk/*.csv")
    for file in files:
        df_temp = pd.read_csv(file)
        df_temp["domain"] = df_temp["prompt_id"].apply(lambda x: id2domain.get(x, "NA"))
        df_temp["query"] = df_temp["prompt_id"].apply(lambda x: id2query.get(x, "NA"))
        # remove "annotation" column if exists
        if "annotation" in df_temp.columns:
            df_temp = df_temp.drop(columns=["annotation"])
        df_temp.to_csv(file, index=False)
        print(df_temp)

def get_responses_open_source_models():
    df = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    id2domain = dict(zip(df["prompt_id"], df["high_risk_label"]))
    id2query = dict(zip(df["prompt_id"], df["content"]))
    prompt_ids_to_keep = set(df["prompt_id"].tolist())

    files = glob.glob("generated_responses/info_seeking_data_for_annotation_total_3000-Llama-3.1-70B-Instruct_*.csv")
    df_all = pd.DataFrame()
    for file in files:
        df_temp = pd.read_csv(file)
        # change "_elisa" in column prompt_id to "_ses"
        df_temp["prompt_id"] = df_temp["prompt_id"].str.replace("_elisa", "_ses")
        print(df_temp)
        df_all = pd.concat([df_all, df_temp], ignore_index=True)
    df_all = df_all[df_all["prompt_id"].isin(prompt_ids_to_keep)]
    df_all["domain"] = df_all["prompt_id"].apply(lambda x: id2domain.get(x, "NA"))
    df_all["query"] = df_all["prompt_id"].apply(lambda x: id2query.get(x, "NA"))
    df_all.drop_duplicates(subset=["prompt_id"], keep="last", inplace=True)
    df_all = df_all[df_all["domain"]!="Other"]
    print(df_all)
    # df_all.to_csv("generated_responses/responses_Llama-3.1-70B-Instruct.csv", index=False)

def unite_all_annotations(category=None):

    c=0

    files = glob.glob("/data1/shared_datasets/project-info-seeking/user_conversations/datasets_filtered/*_history.csv")
    for file in files:

        df_temp = pd.read_csv(file)
        filename = file.split("/")[-1]
        info_seeking = pd.read_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/predictions_infoseek/preds_{filename}")
        id2label = dict(zip(info_seeking["prompt_id"], info_seeking["prediction"]))
        df_temp["info_seeking_label"] = df_temp["prompt_id"].apply(lambda x: id2label.get(x, "NA"))
        df_temp = df_temp[df_temp["info_seeking_label"]=="information seeking"]
        df_temp = df_temp.drop(columns=["info_seeking_label"])
        high_risk = pd.read_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/predictions_highrisk/preds_{filename}")
        id2label = dict(zip(high_risk["prompt_id"], high_risk["prediction"]))
        df_temp["high_risk_label"] = df_temp["prompt_id"].apply(lambda x: id2label.get(x, "NA"))
        if category == "Other":
             df_temp = df_temp[df_temp["high_risk_label"]=="Other"]
        else:
            df_temp = df_temp[df_temp["high_risk_label"]!="Other"]

        infoneeds = pd.read_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/predictions_factual_analytical/preds_{filename}")
        id2label = dict(zip(infoneeds["prompt_id"], infoneeds["prediction"]))
        df_temp["factual_analytical_label"] = df_temp["prompt_id"].apply(lambda x: id2label.get(x, "NA"))
        print(df_temp.columns)
        print(df_temp["factual_analytical_label"].value_counts())
        df_temp = df_temp.drop(columns=["conversation_id"])
        if category == "Other":
            filename = filename.replace("_history.csv", "_other.csv")
        df_temp.drop(columns=["high_risk_label"], inplace=True)
        print(df_temp)
        df_temp.to_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/WildINFOSEEK/{filename}", index=False)
    print(c)

def remove_duplicates_gemini():
    files = glob.glob("./data/responses_with_search/claude-sonnet-4-6_forced_search*.csv")
    df_all = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, on_bad_lines='skip', engine='python')
        df_all = pd.concat([df_all, df], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["prompt_id"], keep="last")
    print(df_all)
    df_all.to_csv("/data1/shared_datasets/project-info-seeking/generated_answers_for_queries/with_forced_search/claude-sonnet-4-6_forced_search_all.csv", index=False)

def news_domains():
    df = pd.read_csv("./data/domains/domains_source_types.csv")
    print(df.source_type.value_counts())
    df = df[df["source_type"].isin(["Encyclopedic", "News", "Archival", "Scientific"])]
    df.rename(columns={"Unnamed: 0":"source"}, inplace=True)
    f = open("./data/domains/info_domains.txt", "w")
    for source in df.source.unique():
        f.write(source+"\n")
    f.close()

def read_responses():
    df=pd.read_csv("/data1/shared_datasets/project-info-seeking/generated_answers_for_queries/responses_no_search/claude-sonnet-4-6_2026-04-20_15-21-52.csv")
    # print row that contains string "is Mao a cunt"
    df_filtered = df[df['query'].str.contains("how to get the firefox key4.db encryption key", na=False)]
    print(df_filtered["query"].tolist())
    print(df_filtered.response.tolist()[0])

def get_test_set_criterion():
    query_response = pd.read_csv("/data1/shared_datasets/project-info-seeking/generated_answers_for_queries/responses_no_search/claude-sonnet-4-6_2026-04-20_15-21-52.csv")
    df = pd.read_csv("safety_eval/claude-sonnet-4-6_2026-04-20_15-21-52_eval_20260512_161754.csv")
    # draw 10 prompt_ids where column "anthropomorphism" == 0 and 10 prompt_ids where column "anthropomorphism" == 1
    df_0 = df[df["anthropomorphism"]==0].sample(n=10, random_state=50)
    df_1 = df[df["anthropomorphism"]==1].sample(n=10, random_state=50)
    df_final = pd.concat([df_0, df_1], ignore_index=True).drop(columns=["response"])
    df_final = df_final.merge(query_response[["prompt_id", "query", "response"]], on="prompt_id", how="left")
    df_final = df_final[["prompt_id", "domain", "query", "response"]]
    print(df_final)
    df_final.sample(20, random_state=50).to_csv("data/annotations/test_set_for_anthropomorphism.csv", index=False)

def merge_joes_annotations():
    df = pd.read_csv("data/annotations/joe_annotation_by_criterion_annotation_round2.csv")
    # create a column "label" where the value comes from "joe_label_new" if "joe_label_new" is not NA, otherwise from "joe_label"
    df["label"] = df.apply(lambda row: row["joe_label_new"] if pd.notna(row["joe_label_new"]) else row["joe_label"], axis=1)
    df = df[["prompt_id", "domain", "response_text", "criterion", "label"]]
    print(df)
    df.to_csv("data/annotations/joe_annotation_by_criterion_annotation_round2.csv", index=False)

def compute_agreement_between_annotators(labels_annotator1="tani_revisited", labels_annotator2="joe_revisited"):
    df = pd.read_csv("data/annotations/agreement_annotation_by_criterion_annotation_round2.csv")
    df["criterion"]= df["criterion"].str.lower()
    annotator1 = df[labels_annotator1].tolist()
    annotator2 = df[labels_annotator2].tolist()
    merged = pd.DataFrame({"a1": annotator1,
                           "a2": annotator2}).dropna()
    if merged["a1"].nunique() < 2 or merged["a2"].nunique() < 2:
        print("Cohen's Kappa overall: skipped (only one class present)")
    else:
        kappa = cohen_kappa_score(merged["a1"].tolist(), merged["a2"].tolist())
        print(f"Cohen's Kappa overall: {kappa}")

    criteria = df["criterion"].unique()
    for criterion in criteria:
        tmp = df[df["criterion"]==criterion]
        merged = pd.DataFrame({"a1": tmp[labels_annotator1].tolist(),
                               "a2": tmp[labels_annotator2].tolist()}).dropna()
        if merged["a1"].nunique() < 2 or merged["a2"].nunique() < 2:
            print(f"Cohen's Kappa for {criterion}: skipped (only one class present)")
            continue
        kappa = cohen_kappa_score(merged["a1"].tolist(), merged["a2"].tolist())
        print(f"Cohen's Kappa for {criterion}: {kappa}")

def add_query_domain_csv():
    # filename = "gpt-5.4_forced_search_tool_2026-04-28_10-58-09.csv"
    filename = "meta-llama-Llama-3.3-70B-Instruct.csv"
    # df = pd.read_csv(f"/data1/shared_datasets/project-info-seeking/generated_answers_for_queries/with_forced_search/{filename}")
    df = pd.read_csv(f"/home/ceron/info-seeking-output-analysis-private/generated_responses/hf_models/{filename}")
    
    if "gpt-5.4_forced_search_tool" in filename: 
    # remove all links from response column which are in this format [link text](url)
        df["response"] = df["response"].str.replace(r"\[.*?\]\(.*?\)", "", regex=True).str.replace("()", "", regex=False)

    df_info = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    id2domain = dict(zip(df_info["prompt_id"], df_info["high_risk_label"]))
    df["domain"] = df["prompt_id"].apply(lambda x: id2domain.get(x, "NA"))
    id2query = dict(zip(df_info["prompt_id"], df_info["content"]))
    df["query"] = df["prompt_id"].apply(lambda x: id2query.get(x, "NA"))
    df.to_csv(f"/home/ceron/info-seeking-output-analysis-private/generated_responses/hf_models/{filename}", index=False)

def count_unique_domains_per_model(filename="data/domains/prompt_id_links_clsf.json"):
    annotations = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    prompt_id_to_domain = dict(zip(annotations["prompt_id"], annotations["annotation"]))
    with open(filename, "r") as f:
        data = json.load(f)
    results = {"Factual": defaultdict(set), "Analytical": defaultdict(set)}
    for model in data:
        for prompt_id in data[model]:
            urls = data[model][prompt_id]
            for url, url_data in urls.items():
                if prompt_id_to_domain[prompt_id] == "Factual":
                    results["Factual"][model].add(url)
                elif prompt_id_to_domain[prompt_id] == "Analytical":
                    results["Analytical"][model].add(url)
                else:
                    print(f"Warning: prompt_id {prompt_id} has unknown domain {prompt_id_to_domain[prompt_id]}")
    # print the number of unique domains per model and per category
    for category in results:
        print(f"Category: {category}")
        for model in results[category]:
            print(f"  Model: {model}, Unique domains: {len(results[category][model])}")
    return results

def count_total_domains_per_model(filename="data/domains/prompt_id_links_clsf.json"):
    annotations = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    prompt_id_to_domain = dict(zip(annotations["prompt_id"], annotations["annotation"]))
    with open(filename, "r") as f:
        data = json.load(f)
    results = {"Factual": defaultdict(int), "Analytical": defaultdict(int)}
    for model in data:
        for prompt_id in data[model]:
            urls = data[model][prompt_id]
            for url, url_data in urls.items():
                if prompt_id_to_domain[prompt_id] == "Factual":
                    results["Factual"][model]+= urls[url]["count"]      
                elif prompt_id_to_domain[prompt_id] == "Analytical":
                    results["Analytical"][model]+= urls[url]["count"]
                else:
                    print(f"Warning: prompt_id {prompt_id} has unknown domain {prompt_id_to_domain[prompt_id]}")
    # print the number of unique domains per model and per category
    for category in results:
        print(f"Category: {category}")
        for model in results[category]:
            print(f"  Model: {model}, Total N. domains: {results[category][model]}")
    return results


def compute_pielou_evenness(filename="data/domains/prompt_id_links_clsf.json"):
    annotations = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")
    prompt_id_to_annotation = dict(zip(annotations["prompt_id"], annotations["annotation"]))
    with open(filename, "r") as f:
        data = json.load(f)

    # domain frequency per (annotation_type, model)
    freq = {"Factual": defaultdict(lambda: defaultdict(int)),
            "Analytical": defaultdict(lambda: defaultdict(int))}
    for model in data:
        for prompt_id, urls in data[model].items():
            ann = prompt_id_to_annotation.get(prompt_id)
            if ann not in freq:
                continue
            for url in urls:
                print(urls[url]["count"], url)
                freq[ann][model][url] += urls[url]["count"]

    results = {}
    for ann_type, models in freq.items():
        results[ann_type] = {}
        for model, domain_counts in models.items():
            counts = np.array(list(domain_counts.values()), dtype=float)
            s = len(counts)
            print(s)
            if s <= 1:
                results[ann_type][model] = float("nan")
                continue
            p = counts / counts.sum()
            h = -np.sum(p * np.log(p))
            results[ann_type][model] = h / math.log(s)

    for ann_type in results:
        print(f"Annotation: {ann_type}")
        for model, j in results[ann_type].items():
            print(f"  {model}: J = {j:.4f}")
    return results


def process_urls_for_scraper():
    filename = "data/domains/prompt_id_links_clsf.json"
    with open(filename, "r") as f:
        data = json.load(f)
    urltotype = {}
    for model in data:
        for prompt_id in data[model]:
            urls = data[model][prompt_id]
            for url, url_data in urls.items():
                type = url_data["type"]
                urltotype[url] = type
    df = pd.DataFrame(list(urltotype.items()), columns=["url", "type"])
    df = df[df["type"].isin(["News", "Encyclopedic", "Archival", "Scientific"])]
    already_processed_urls = "data/domains/info_domains.txt"
    with open(already_processed_urls, "r") as f:
        processed_urls = set(line.strip() for line in f)
    # remove from df the rows where the url is in processed_urls. 
    df = df[~df["url"].isin(processed_urls)]
    new_urls = "data/domains/info_domains2.txt"
    with open(new_urls, "w") as f:
        for url in df["url"].unique():
            f.write(url+"\n")
    f.close()

def match_wildseek_history():
    df_history = pd.DataFrame()
    files = glob.glob("/data1/shared_datasets/project-info-seeking/user_conversations/datasets_filtered/*_history.csv")
    for file in files:
        df = pd.read_csv(file)
        if "ses" in file.lower():
            df["prompt_id"] = df["prompt_id"].str.lower()
        df_history = pd.concat([df_history, df], ignore_index=True)
    wildseek = pd.read_csv("data_prompts/man-annotated-train-high-risk.csv")[2000:]
    id2prompt = dict(zip(wildseek["prompt_id"], wildseek["content"]))
    for i, id in enumerate(id2prompt.keys()):
        if id in df_history["prompt_id"].values:
            if id2prompt[id].strip() != df_history[df_history["prompt_id"]==id]["content"].values[0].strip(): 
                print(f"Processing prompt_id {id} (index {i})")
                print(f"history content: {df_history[df_history['prompt_id']==id]['content'].values[0]}\nexpected content: {id2prompt[id]}")
        else:
            print(f"\n\nNo match found for prompt_id {id}, index {i}")

if __name__ == "__main__":
    # get_info_seeking_queries()
    # fix_col_history()
    # get_sample()
    # for dataset in ["en_WildChat", "en_elisa"]:
    #     all_prompts_predicted(dataset)
    # process_stakes_dataset()
    # get_sample_info_seeking()
    # match_preds_queries()
    # llm_responses()
    # process_orcas()
    # get_marcos()
    # natural_questions_dataset()
    # add_more_data_test_set()
    # get_training_sample()
    # process_validated_training_set()
    # deduplicate_train_test_datasets()
    # add_unique_id_col_training_datasets()   
    # gather_all_datasets()
    # read_dataset()
    # map_queries_preds("en_ShareGPT")
    # prepare_data_for_annotation()
    # calculate_cohens_kappa()
    # join_infoseek_with_validation()
    # correct_prompt_ids_lmsys_chat_1m()
    # remove_duplicates_lmsys_chat_1m()
    # gather_infoseek_llm_train_high_risk()
    # join_infoseek_with_high_risk()
    # create_csv_samples_for_annotations()
    # get_factual_queries()
    # annotate_generated_responses()
    # get_responses_open_source_models()
    # unite_all_annotations(category="Other")
    # unite_all_annotations()
    # remove_duplicates_gemini()
    # news_domains()
    # read_responses()
    # get_test_set_criterion()
    # merge_joes_annotations()
    # compute_agreement_between_annotators("tani_label", "joe_label")
    # compute_agreement_between_annotators()
    # add_query_domain_csv()
    # process_urls_for_scraper()
    # count_unique_domains_per_model()
    # compute_pielou_evenness()
    # count_total_domains_per_model()
    # match_wildseek_history()
    pass
