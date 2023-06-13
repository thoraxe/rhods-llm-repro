import pandas as pd
import json
import torch
from datasets import load_dataset

import gc
gc.collect()
torch.cuda.empty_cache()

from transformers import (
    AutoTokenizer, 
    BloomModel, 
    AutoModel, 
    BloomForCausalLM, 
    TrainingArguments, 
    Trainer,
    pipeline)

#########################

print("Loading and prepping data...")
json1_path = "event1.json"
with open(json1_path) as file:
    input_data_1 = file.read()

summary1_path = "summary1.txt"
with open(summary1_path, "r") as file:
    summary_1 = file.read()

json2_path = "event2.json"
with open(json2_path) as file:
    input_data_2 = file.read()

summary2_path = "summary2.txt"
with open(summary2_path, "r") as file:
    summary_2 = file.read()

json3_path = "event3.json"
with open(json3_path) as file:
    input_data_3 = file.read()

summary3_path = "summary3.txt"
with open(summary3_path, "r") as file:
    summary_3 = file.read()

########################

df_train = pd.DataFrame()
df_train["input"] = [input_data_1, input_data_2, input_data_3]
df_train["output"] = [summary_1, summary_2, summary_3]

print(df_train.head())

df_train.to_json("training.jsonl", orient="records", lines=True)

#########################

print("Setting up model training...")
SEED_VALUE = 42
MODEL_NAME = "bigscience/bloom-560m"
BATCH_SIZE = 2
EPOCHS = 5

task_designator = "Summary:"
context_length = 2048
padding = "max_length"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BloomForCausalLM.from_pretrained(MODEL_NAME)

print("Setting training arguments...")
args = TrainingArguments(
    output_dir= f"fine-tuned/{MODEL_NAME}_summarizer_{EPOCHS}_epochs",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="steps",
    eval_steps=5000,
    logging_steps=1000,
    num_train_epochs=EPOCHS,
    learning_rate=5e-6,
    fp16=True,
    save_strategy="epoch",
    save_total_limit=2
)

print("Loading datasets...")
data = load_dataset("json", data_files={"train":["training.jsonl"]})

def tokenize(element):
    ip = ""
    for ele in element["input"]:
        ip += str(ele)
    print(ip)
    text = "Data: " + ip + "\n" + task_designator + " " + element["output"] + tokenizer.eos_token
    output = tokenizer(
        text, 
        truncation=True,
        padding=padding,
        max_length=context_length,
        )
    
    labels = output["input_ids"].copy()
    labels = [-100 if ele == tokenizer.pad_token_id else ele for ele in labels]
    output["labels"] = labels
    return output

print("Mapping tokenized data...")
tokenized_datasets = data.map(
    tokenize, remove_columns=data["train"].column_names
)

print("Setting up trainer...")
#trainer = Trainer(
#    model=model,
#    tokenizer=tokenizer,
#    args=args,
#    train_dataset=tokenized_datasets["train"],
#    eval_dataset=tokenized_datasets["validation"],
#)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized_datasets["train"]
)

print("Training...")
trainer.train()

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#checkpoint = f"fine-tuned/bloom_1b1_summarizer_{EPOCHS}_epochs/checkpoint-20"

