# =========================================================
# LEGAL TURKISH BERT – 5 RUN FULL TRAINING PIPELINE
# =========================================================

import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# 1️⃣ DOSYA YOLLARI
# =========================================================
FILES = {
    "train": "dataset-v1-train.csv",
    "test": "datase-v1-test.csv"   # typo bilinçli korunuyor
}

BASE_MODEL_NAME = "msbayindir/legal-turkish-bert-base-cased"
BASE_OUTPUT_DIR = "./trained-models"
NUM_RUNS = 5

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2️⃣ DATASET
# =========================================================
dataset = load_dataset("csv", data_files=FILES)

# =========================================================
# 3️⃣ TOKENIZER
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256
    )

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# =========================================================
# 4️⃣ LABEL MAP
# =========================================================
id2label = {0: "YUKSEK_RISK", 1: "ORTA_RISK", 2: "RISKSIZ"}
label2id = {v: k for k, v in id2label.items()}

# =========================================================
# 5️⃣ DEVICE
# =========================================================
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"🖥️ Kullanılan cihaz: {device.upper()}")

# =========================================================
# 6️⃣ METRICS
# =========================================================
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_metric.compute(
        predictions=preds,
        references=labels
    )["accuracy"]

    f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="weighted"
    )["f1"]

    return {"accuracy": acc, "f1": f1}

# =========================================================
# 7️⃣ DATA COLLATOR
# =========================================================
collator = DataCollatorWithPadding(tokenizer)

# =========================================================
# 8️⃣ 5 RUN TRAINING
# =========================================================
all_run_metrics = []

for run_id in range(1, NUM_RUNS + 1):
    print(f"\n🚀 RUN {run_id} BAŞLIYOR\n")

    run_dir = f"{BASE_OUTPUT_DIR}/run-{run_id}"
    final_model_dir = f"{run_dir}/final-model"

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=42 + run_id,
        logging_dir=f"{run_dir}/logs",
        report_to=["tensorboard"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    all_run_metrics.append({
        "run": run_id,
        "accuracy": eval_metrics["eval_accuracy"],
        "f1": eval_metrics["eval_f1"],
        "model_path": final_model_dir
    })

    print(f"✅ RUN {run_id} TAMAMLANDI")

# =========================================================
# 9️⃣ METRICS KAYDET
# =========================================================
summary_path = f"{BASE_OUTPUT_DIR}/summary_metrics.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(all_run_metrics, f, indent=4, ensure_ascii=False)

print(f"\n📁 Metrics kaydedildi → {summary_path}")

# =========================================================
# 🔟 EN İYİ MODEL SEÇ
# =========================================================
best_run = max(all_run_metrics, key=lambda x: x["f1"])
BEST_MODEL_PATH = best_run["model_path"]

print("\n🏆 EN İYİ MODEL")
print(best_run)

# =========================================================
# 1️⃣1️⃣ CONFUSION MATRIX & CLASS REPORT
# =========================================================
best_model = AutoModelForSequenceClassification.from_pretrained(
    BEST_MODEL_PATH
).to(device)

trainer.model = best_model
preds = trainer.predict(tokenized_datasets["test"])

y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print("\n📊 CONFUSION MATRIX (BEST MODEL)")
print(confusion_matrix(y_true, y_pred))

print("\n📄 CLASSIFICATION REPORT (BEST MODEL)")
print(classification_report(
    y_true,
    y_pred,
    target_names=["YUKSEK_RISK", "ORTA_RISK", "RISKSIZ"]
))

# =========================================================
# 1️⃣2️⃣ ENSEMBLE (5 MODEL – MAJORITY VOTE)
# =========================================================
all_predictions = []

for run in all_run_metrics:
    model = AutoModelForSequenceClassification.from_pretrained(
        run["model_path"]
    ).to(device)

    trainer.model = model
    p = trainer.predict(tokenized_datasets["test"]).predictions
    all_predictions.append(np.argmax(p, axis=1))

all_predictions = np.stack(all_predictions)  # (5, N)

ensemble_preds = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(),
    axis=0,
    arr=all_predictions
)

print("\n🧠 ENSEMBLE CONFUSION MATRIX")
print(confusion_matrix(y_true, ensemble_preds))

print("\n🧠 ENSEMBLE CLASSIFICATION REPORT")
print(classification_report(
    y_true,
    ensemble_preds,
    target_names=["YUKSEK_RISK", "ORTA_RISK", "RISKSIZ"]
))

print("\n✅ TÜM PIPELINE TAMAMLANDI")
