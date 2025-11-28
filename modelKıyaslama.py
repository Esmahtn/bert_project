import sys
import os
import shutil
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import evaluate

# ⚠️ ÖN KOŞULLAR: Veri setleri ve kütüphaneler yüklü olmalıdır.

# 1️⃣ Odaklanılacak Modeller Listesi
MODEL_LIST_FOCUSED_V3 = [
    {"name": "Legal-BERT (V3)", "path": "msbayindir/legal-turkish-bert-base-cased", "alias": "msb_legal_v3"},
    {"name": "Law-EQA (V3)", "path": "yeniguno/turkish-law-eqa-bert-finetuned", "alias": "yg_eqa_v3"},
]

# 2️⃣ Veri Yükleme ve Ön İşleme
files = {
    "train": "dataset-v1-train.csv", 
    "test": "datase-v1-test.csv"
}
try:
    dataset = load_dataset("csv", data_files=files)
except FileNotFoundError:
    print("🚨 HATA: Veri setleri bulunamadı.")
    sys.exit(1)

# Etiket Haritası
id2label = {0: "YUKSEK_RISK", 1: "ORTA_RISK", 2: "RISKSIZ"}
label2id = {"YUKSEK_RISK": 0, "ORTA_RISK": 1, "RISKSIZ": 2}

# Metrikler
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    """Eğitim metriklerini hesaplar: Accuracy ve Ağırlıklı F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = acc.compute(predictions=preds, references=labels)["accuracy"]
    f1w = f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1_weighted": f1w}

# Cihaz Ayarı
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def train_and_evaluate_model_v3(model_info, dataset):
    """Belirtilen modeli optimize edilmiş ayarlar ile fine-tuning eder."""
    model_path = model_info["path"]
    model_alias = model_info["alias"]
    output_dir = f"./results-{model_alias}"
    save_path = f"./trained-models/{model_alias}"

    print(f"\n=======================================================")
    print(f"🚀 BAŞLANIYOR: {model_info['name']} - ÜÇÜNCÜ TUR AYARLARI")
    print(f"=======================================================")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    # Boyut uyumsuzluğu düzeltmesi
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True 
    )
    model.to(device)
    
    # ⚙️ ÜÇÜNCÜ TUR AYARLARI
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,               # Sabit tutuldu
        per_device_train_batch_size=16,   # 12'den 16'ya yükseltildi
        per_device_eval_batch_size=16,
        learning_rate=3e-5,               # 2e-5'ten 3e-5'e yükseltildi
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=["none"]
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Model kaydedildi: {save_path}")

    evaluation_results = trainer.evaluate()
    shutil.rmtree(output_dir, ignore_errors=True)
    
    return evaluation_results

# 3️⃣ Ana Çalıştırma Bloğu
if __name__ == "__main__":
    
    all_results = {}
    os.makedirs("./trained-models", exist_ok=True)

    for model_info in MODEL_LIST_FOCUSED_V3:
        try:
            results = train_and_evaluate_model_v3(model_info, dataset)
            all_results[model_info["name"]] = {
                "Accuracy": results.get("eval_accuracy", 0),
                "F1_Weighted": results.get("eval_f1_weighted", 0),
                "Loss": results.get("eval_loss", float('inf'))
            }
        except Exception as e:
            print(f"❌ KRİTİK HATA OLUŞTU ({model_info['name']}): {e}")
            all_results[model_info["name"]] = {"Error": str(e)}

    # 4️⃣ Üçüncü Tur Sonuçlarını Yazdırma
    print("\n\n=======================================================")
    print("🏆 ÜÇÜNCÜ TUR OPTİMİZASYON KARŞILAŞTIRMASI")
    print(f"AYARLAR: Epochs=8, Batch=16, LR=3e-5")
    print("=======================================================")
    print(f"{'Model Adı':<25} | {'Accuracy':<10} | {'F1 Weighted':<15} | {'Loss':<10}")
    print("-" * 65)

    best_f1 = -1
    best_model = ""

    for name, metrics in all_results.items():
        if "Error" in metrics:
            print(f"{name:<25} | {'HATA':<10} | {metrics['Error']}")
            continue
            
        acc_str = f"{metrics['Accuracy']:.4f}"
        f1_str = f"{metrics['F1_Weighted']:.4f}"
        loss_str = f"{metrics['Loss']:.4f}"
        
        print(f"{name:<25} | {acc_str:<10} | {f1_str:<15} | {loss_str:<10}")

        if metrics['F1_Weighted'] > best_f1:
            best_f1 = metrics['F1_Weighted']
            best_model = name

    print("\n-------------------------------------------------------")
    print(f"🥇 EN İYİ MODEL (V3): {best_model} ({best_f1:.4f})")
    print("-------------------------------------------------------")