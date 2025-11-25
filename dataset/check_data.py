import pandas as pd

# Dosya yollarını kendine göre düzenle
train_path = 'final_train_dataset.csv' # Senin birleştirdiğin dosya
test_path = '/Users/pelinsusaglam/Desktop/dataset_duzenle/dataset-v1(duzenlenecek)/test_dataset.csv'

try:
    # Pandas ile okumayı dene
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print("✅ Dosyalar başarıyla okundu!")
    print("-" * 30)

    # 1. Sütun İsimlerini Kontrol Et
    print(f"Train Sütunları: {df_train.columns.tolist()}")
    
    # 2. Boş Veri Var mı? (Null Check)
    bos_train = df_train.isnull().sum().sum()
    print(f"Train setindeki boş hücre sayısı: {bos_train} (0 olmalı)")
    
    # 3. Örnek Bir Satıra Bak (Virgül sorunu var mı?)
    print("-" * 30)
    print("Örnek Veri Kontrolü:")
    print(f"Metin: {df_train.iloc[0, 0]}") # İlk satırın metni
    print(f"Etiket: {df_train.iloc[0, 1]}") # İlk satırın etiketi
    
    # 4. Etiketlerin Tipini Kontrol Et
    # Etiketler 'int' (tam sayı) olmalı
    print("-" * 30)
    print(f"Etiket Veri Tipi: {df_train.dtypes[1]}") 
    
    if bos_train == 0:
        print("\n🚀 SONUÇ: Veri seti BERT eğitimi için teknik olarak temiz görünüyor.")
    else:
        print("\n⚠️ DİKKAT: Veri setinde boş satırlar var, temizlenmeli.")

except Exception as e:
    print("\n❌ HATA: Dosya okunurken sorun oluştu.")
    print("Olası sebep: Cümle içindeki virgüller sütunları kaydırmış olabilir.")
    print(f"Hata detayı: {e}")