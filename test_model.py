from transformers import pipeline
import torch # Gerekli kütüphaneyi ekliyoruz

# 1. Pipeline'ı Yükle
model_path = "./final-legal-bert-risk-model"
risk_classifier = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    # BURAYI DEĞİŞTİRİN: GPU'dan (-1) CPU'ya geçiyoruz
    device=-1  
)

# 2. Sınıflandırmak İstediğiniz Örnek Metinler (Aynı Kalır)
texts_to_analyze = [
    "Sözleşmenin bu maddesi, taraflardan birinin tek taraflı fesih hakkını herhangi bir bildirim süresi olmaksızın kullanabileceğini belirtir.", 
    "İş bu sözleşme, tarafların karşılıklı iyi niyet çerçevesinde mutabık kaldığı şekilde, [TARİH] tarihinde yürürlüğe girer.", 
    "Tedarikçinin iflası durumunda sözleşme kendiliğinden feshedilecektir." 
]

print("\n--- Sınıflandırma Sonuçları ---")

for text in texts_to_analyze:
    # Model yüklendikten sonra tahmin başlar
    result = risk_classifier(text)[0]
    
    label_turkish = result['label'].replace('_', ' ') 
    
    print(f"\nMetin: \"{text[:60]}...\"")
    print(f"  Tahmin Edilen Risk: {label_turkish}")
    print(f"  Güven Skoru: {result['score']:.4f}")