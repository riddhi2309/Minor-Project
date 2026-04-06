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

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("data/cybersecurity_intent_dataset.csv")

df["text"] = df["text"].fillna("").astype(str)

labels = sorted(df["label"].unique())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

df["label"] = df["label"].map(label2id)

# =============================
# TRAIN / VAL / TEST SPLIT
# =============================
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

# =============================
# TOKENIZER
# =============================
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

# =============================
# MODEL
# =============================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# =============================
# METRICS
# =============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# =============================
# TRAINING CONFIG
# =============================
training_args = TrainingArguments(
    output_dir="models/intent_classifier",

    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    learning_rate=2e-5,
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

print("Final evaluation:")
print(trainer.evaluate(test_dataset))

trainer.save_model("models/intent_classifier")
tokenizer.save_pretrained("models/intent_classifier")