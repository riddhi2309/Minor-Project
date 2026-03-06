import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL = "roberta-base"

df = pd.read_csv("data/jailbreak.csv")

print("Columns:", df.columns)

# Convert conversation → text
df["text"] = df["conversation"].fillna("").astype(str)

# Keep only required columns
df = df[["text", "label"]]

# Normalize labels
df["label"] = df["label"].astype(str).str.strip().str.upper()

label_map = {
    "BENIGN_SAFE": 0,
    "SAFE": 0,
    "BENIGN": 0,
    "NORMAL": 0,
    "NON_JAILBREAK": 0,

    "JAILBREAK": 1,
    "JAILBREAK_ATTACK": 1
}
print("Unique labels:", df["label"].unique())
df["label"] = df["label"].map(label_map)

if df["label"].isnull().sum() > 0:
    print("Unmapped labels detected:")
    print(df[df["label"].isnull()].head())
    raise ValueError("Fix label mapping above")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

tokenizer = AutoTokenizer.from_pretrained(MODEL)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

training_args = TrainingArguments(
    output_dir="models/jailbreaking",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=200,
    do_train=True,
    do_eval=True,
    report_to=[]
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

print(trainer.evaluate(test_dataset))

trainer.save_model("models/jailbreaking_detector")
tokenizer.save_pretrained("models/jailbreaking_detector")