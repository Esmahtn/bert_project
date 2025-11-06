from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch, evaluate, numpy as np

# 1️⃣ Veri yükle
dataset = load_dataset("csv", data_files={"train": "data.csv", "test": "data.csv"})

# 2️⃣ Tokenizer (BERT Large)
model_name = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# 3️⃣ Model (label haritası)
id2label = {0: "RISKLI", 1: "RISKSIZ", 2: "BELIRSIZ"}
label2id = {"RISKLI": 0, "RISKSIZ": 1, "BELIRSIZ": 2}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

if torch.cuda.is_available():
    model.to("cuda")

# 4️⃣ Metrikler
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = acc.compute(predictions=preds, references=labels)["accuracy"]
    f1w = f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1w}

# 5️⃣ Eğitim ayarları
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    fp16=True,
    report_to="none"
)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./bert-large-risk-model")
tokenizer.save_pretrained("./bert-large-risk-model")

print("✅ Eğitim tamamlandı ve model kaydedildi!")
