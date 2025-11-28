import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# 🚨 GÜNCEL KLASÖR YOLU: İlk turda eğitilen 5-epoch Law-EQA modeli.
BEST_MODEL_PATH = "./trained-models/yg_eqa"

# Etiket Haritası
id2label = {0: "YUKSEK_RISK", 1: "ORTA_RISK", 2: "RISKSIZ"}

# 1. Kalitatif Test Edilecek Sözleşme Maddeleri (Aynı Örnekler)
test_sentences = [
    # YÜKSEK RİSK Örnekleri
    {"text": "Bu sözleşme, karşı tarafın yazılı izni olmaksızın tarafımızca herhangi bir sebeple derhal ve tek taraflı olarak feshedilebilir.", "expected_label": "YUKSEK_RISK"},
    {"text": "Tazminat taleplerinde, şirketin sorumluluğu yalnızca sözleşme bedelinin %5'i ile sınırlıdır ve bu sınır aşılamaz.", "expected_label": "YUKSEK_RISK"},
    
    # ORTA RİSK Örnekleri
    {"text": "Gizlilik süresi, sözleşme sona erdikten sonra 1 yıl olarak belirlenmiştir, ancak bu süre özel bir durum halinde uzatılabilir.", "expected_label": "ORTA_RISK"},
    {"text": "Gecikme durumunda cezai şart uygulanır, ancak karşı tarafın mücbir sebep ispatlaması halinde ceza kaldırılır.", "expected_label": "ORTA_RISK"},
    
    # RİSKSİZ Örnekleri
    {"text": "Taraflar arasındaki tebligatlar, PTT aracılığı ile madde 2'de belirtilen resmi adreslere gönderilecektir.", "expected_label": "RISKSIZ"},
    {"text": "İşbu sözleşme 10 maddeden oluşmaktadır ve tüm maddeler taraflarca tam olarak okunmuş ve kabul edilmiştir.", "expected_label": "RISKSIZ"},
]

def predict_risk_level(text):
    """Verilen metin için risk seviyesini ve güven skorunu tahmin eder."""
    
    # Model ve Tokenizer yükleniyor
    try:
        tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_PATH)
    except Exception as e:
        # Eğer model yüklenemezse hata ver
        return f"Model Yükleme Hatası: {BEST_MODEL_PATH} yolunda model bulunamadı veya yüklenemedi. Hata: {e}"

    # Metnin token haline getirilmesi
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
    # Tahmin yapılması
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Logitleri olasılığa çevirme (Softmax)
    probabilities = torch.softmax(logits, dim=1)
    
    # En yüksek olasılığa sahip sınıfı bulma
    predicted_class_id = torch.argmax(probabilities).item()
    predicted_label = id2label[predicted_class_id]
    
    # Güven Skorunu hesaplama (%)
    confidence_score = probabilities[0][predicted_class_id].item() * 100
    
    return predicted_label, confidence_score

def print_results(results):
    """Tahmin sonuçlarını rapor formatında yazdırır."""
    
    print("\n" + "="*90)
    print("🔬 TABLO 4: LAW-EQA (5 Epoch / yg_eqa) KALİTATİF ANALİZİ")
    print(f"📉 Model Skoru: F1 ~0.8644 (İlk Tur Sonucu)")
    print("="*90)
    
    header = f"{'Madde Tanımı':<50} | {'Beklenen Etiket':<15} | {'Model Tahmini':<15} | {'Güven Skoru':<10}"
    print(header)
    print("-" * 90)
    
    for item in results:
        madde_tanimi = item['text'][:47] + "..." if len(item['text']) > 50 else item['text']
        
        if item['predicted_label'] == item['expected_label']:
            tahmin_str = f"✅ {item['predicted_label']}"
        else:
            tahmin_str = f"❌ {item['predicted_label']}"

        output_line = (
            f"{madde_tanimi:<50} | "
            f"{item['expected_label']:<15} | "
            f"{tahmin_str:<15} | "
            f"{item['confidence_score']:.1f} %"
        )
        print(output_line)
        
    print("-" * 90)


if __name__ == "__main__":
    results = []
    
    print(f"Law-EQA (5 Epoch) modeli yükleniyor ve kalitatif analiz yapılıyor (Yol: {BEST_MODEL_PATH})...")
    
    for sentence in test_sentences:
        prediction_result = predict_risk_level(sentence["text"])
        
        if isinstance(prediction_result, str) and "Hata" in prediction_result:
            print(prediction_result)
            break
            
        predicted_label, confidence_score = prediction_result
        
        results.append({
            "text": sentence["text"],
            "expected_label": sentence["expected_label"],
            "predicted_label": predicted_label,
            "confidence_score": confidence_score
        })
    
    if results:
        print_results(results)