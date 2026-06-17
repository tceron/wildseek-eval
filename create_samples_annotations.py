import json
import pandas as pd
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os

def prepare_data_wildchat(dataset_name="allenai/WildChat"):
    ds = load_dataset(dataset_name)
    sample = []
    for row in tqdm(ds["train"]):
        langs = []
        for i in range(len(row["conversation"])):
            item = row["conversation"][i]
            langs.append(item["language"])
        if set(langs) == {"English"}:
            for i in range(len(row["conversation"])):
                item = row["conversation"][i]
                if item["role"] == "user":
                    sample.append((row["conversation_id"], i, item["content"]))
                # print((row["conversation_id"], i, item["content"]))

    df = pd.DataFrame(sample, columns=["conversation_id", "turn", "content"])
    print(len(df))
    dataset_name = dataset_name.split("/")[1]
    df.to_csv(f"./data/en_{dataset_name}.csv", index=False)


def prepare_data_lmsys(dataset_name="lmsys/lmsys-chat-1m"):
    ds = load_dataset(dataset_name)
    sample = []
    for row in tqdm(ds["train"]):
        if row["language"] == "English":
            for i in range(len(row["conversation"])):
                item = row["conversation"][i]
                if item["role"] == "user":
                    sample.append((row["conversation_id"], i, item["content"]))

    df = pd.DataFrame(sample, columns=["conversation_id", "turn", "content"])
    print(len(df))
    dataset_name = dataset_name.split("/")[1]
    df.to_csv(f"./data/en_{dataset_name}.csv", index=False)

# prepare_data_lmsys("lmsys/lmsys-chat-1m")

def create_history_dataset():
    df = pd.read_csv("/data1/shared_datasets/project-info-seeking/user_conversations/en_lmsys-chat-1m.csv")

    # Ensure rows are ordered within each conversation
    df = df.sort_values(["conversation_id", "turn"]).reset_index(drop=True)

    # Build history consisting only of previous turns in the same conversation
    histories = []
    current_conversation_id = None
    current_history = []

    for _, row in df.iterrows():
        if row["conversation_id"] != current_conversation_id:
            # New conversation: reset history
            current_conversation_id = row["conversation_id"]
            current_history = []

        # Append a copy of the history *before* this turn
        histories.append(list(current_history))

        # Then add the current content to the running history
        current_history.append(row["content"])

    df["conversation_history"] = histories
    df["prompt_id"] = [str(i)+"_lmsys-chat-1m" for i in range(len(df))]
    df.to_csv("/data1/shared_datasets/project-info-seeking/user_conversations/en_lmsys-chat-1m_history.csv", index=False)

# create_history_dataset()

def add_prompt_id_to_history_datasets():
    for dataset in ["lmsys-chat-1m", "WildChat", "ShareGPT", "SES"][3:]:
        df = pd.read_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/en_{dataset}_history.csv")
        prompt_ids = [str(i)+"_"+dataset for i in range(len(df))]
        if "prompt_id" in df.columns:
            df.drop(columns=["prompt_id"], inplace=True)
        df.insert(0, "prompt_id", prompt_ids)   
        df.to_csv(f"/data1/shared_datasets/project-info-seeking/user_conversations/en_{dataset}_history.csv", index=False)
        print(df)


def sample_annotations(filenames):

    if os.path.exists(f"./data/final_annotations/gold_standard_test_set.csv"):
        annotated = pd.read_csv(f"./data/final_annotations/gold_standard_test_set.csv")
        annot_ids = annotated.conversation_id.tolist()
    else:
        annot_ids = []

    sample_df = pd.DataFrame()
    for filename in filenames:
        df = pd.read_csv(f"./data/en_{filename}.csv")
        df.insert(0, "dataset", len(df) * [filename])
        if filename == "elisa":
            df.insert(1, "turn" ,[None] * len(df))
        # sample non repetitive rows from the dataframe
        df = df.drop_duplicates()
        num_sample = 0

        for i in np.random.choice(df.conversation_id.unique(), 100, replace=False):
            if num_sample < 100:
                if i in annot_ids:
                    continue
                else:    
                    tmp = df[df.conversation_id == i]
                    # check if there's "Midjourney" in any of the string of the rows of column "content"
                    if tmp.content.str.contains("Midjourney").any():
                        continue
                    else:
                        sample_df = pd.concat([sample_df, tmp])
                        num_sample += len(tmp)
    print(sample_df)
    sample_df.to_csv(f"./data/annotations/annotations_round4.csv", index=False)

# sample_annotations(["elisa", "ShareGPT"])

def merge_samples():
    df = pd.read_csv("./data/annotations/annotations_lmsys-chat-1m_round2.csv")
    df2 = pd.read_csv("./data/annotations/annotations_WildChat_round2.csv")
    df["dataset"]=len(df)*["lmsys-chat-1m"]
    df2["dataset"]=len(df2)*["WildChat"]
    df = pd.concat([df, df2])
    df.to_csv("./data/annotations/annotations_round2.csv", index=False)


def read_annotated_sharegpt():
    id2task = {}
    with open("/data1/shared_datasets/shareGPT/shareGPT_merged.json", "r") as f:
        for line in f:
            l = json.loads(line)
            id2task[l["id"]] = l["task_type"]

    return id2task

def prepare_data_sharegpt(file_path):
    with open(f"/data1/shared_datasets/shareGPT/{file_path}", "r") as f:
        data = json.load(f)

    sample = []
    for item in data:
        for i in range(len(item["conversations"])):
            if item["conversations"][i]["from"] == "human":
                sample.append((item["id"], i, item["conversations"][i]["value"]))

    df = pd.DataFrame(sample, columns=["conversation_id", "turn", "content"])
    print(df)
    # sample non repetitive rows from the dataframe
    df.to_csv("./data/en_ShareGPT.csv", index=False)

# prepare_data_sharegpt("ShareGPT_V3_unfiltered_cleaned_split.json")

def prepare_elisa_data():
    df = pd.read_csv("./data/survey_language_technologies.csv")

    prompts = []
    
    for _, row in df.iterrows():
        for i in range(1,11):
            if str(row[f"prompt{i}"]) != "nan":
                prompts.append((str(row["id"])+"_"+str(i), row[f"prompt{i}"].rstrip().lstrip()))
    df = pd.DataFrame(prompts, columns=["conversation_id", "content"])
    df.to_csv("./data/en_elisa.csv", index=False)

# prepare_elisa_data()

def check_prompts():
    df = pd.read_csv("./data/annotations_sharegpt.csv")
    for i in df.conversation_id.unique():
        tmp = df[df.conversation_id == i]
        print("id--",i)
        for row in tmp.iterrows():
            print(row[1].turn,row[1].content)
        print("task type--",row[1].task_type)
        print("-------------------------------------------------", "\n")