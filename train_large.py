from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import torch
import evaluate
import numpy as np

# ⚠️ ÖN KOŞULLAR:
# 'dataset-v1-train.csv' ve 'datase-v1-test.csv' dosyaları bu kod ile aynı klasörde olmalıdır.
# Gerekli kütüphaneler (transformers, datasets, evaluate, numpy) yüklü olmalıdır.

# 1️⃣ Veri Yükleme
files = {
    "train": "dataset-v1-train.csv", 
    "test": "datase-v1-test.csv"  # Test dosyasının typo'su korundu.
}
dataset = load_dataset("csv", data_files=files)

# 2️⃣ Tokenizer (Türkçe Legal BERT)
model_name = "msbayindir/legal-turkish-bert-base-cased" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    # Metinleri tokenize et, max uzunluk 256
    return tokenizer(examples["text"], truncation=True, max_length=256)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 3️⃣ Model ve Etiket Haritası (etiket_id.txt'ye göre)
id2label = {0: "YUKSEK_RISK", 1: "ORTA_RISK", 2: "RISKSIZ"}
label2id = {"YUKSEK_RISK": 0, "ORTA_RISK": 1, "RISKSIZ": 2}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# Cihaz Ayarı
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# 4️⃣ Metrikler
acc = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = acc.compute(predictions=preds, references=labels)["accuracy"]
    # F1 skoru, sınıfların dengesizliğinden etkilenmemesi için "weighted" kullanılıyor.
    f1w = f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1w}

# 5️⃣ Eğitim Ayarları
args = TrainingArguments(
    output_dir="./results-legal-bert",
    num_train_epochs=10,              # Güvenli üst limit. En iyi model korunacaktır.
    per_device_train_batch_size=16,   # Base model için yüksek batch size
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,               
    weight_decay=0.01,
    evaluation_strategy="epoch",      
    save_strategy="epoch",
    load_best_model_at_end=True,      # Overfitting riskine karşı en iyi modeli otomatik yükler
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    fp16=torch.cuda.is_available(),   
    logging_dir="./logs",
    report_to=["tensorboard"]
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

print(f"🚀 Eğitim başlıyor: {model_name}. Cihaz: {device.upper()}")
trainer.train()

# 6️⃣ Modeli Kaydet
save_path = "./final-legal-bert-risk-model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Eğitim tamamlandı! Model '{save_path}' klasörüne kaydedildi.")