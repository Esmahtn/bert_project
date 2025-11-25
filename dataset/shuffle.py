import pandas as pd

# 1. Dosyaları Oku
# Eğer senin dosya isimlerin farklıysa parantez içlerini değiştir.
df0 = pd.read_csv('train_label_0.1.csv')
df1 = pd.read_csv('train_label_1.1.csv')
df2 = pd.read_csv('train_label_2.1.csv')

# 2. Hepsini Alt Alta Birleştir (Concat)
tum_veri = pd.concat([df0, df1, df2])

# 3. KARIŞTIRMA (SHUFFLE) - En Önemli Kısım
# frac=1 tüm veriyi alır, random_state=42 her seferinde aynı karmaşıklığı verir
tum_veri_karisik = tum_veri.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Kaydet
tum_veri_karisik.to_csv('final_train_dataset.csv', index=False)

print(f"İşlem Tamam!")
print(f"Toplam Veri Sayısı: {len(tum_veri_karisik)}")
print("İlk 5 satırın etiketlerine bak (Karışık olmalı):")
print(tum_veri_karisik['label'].head(10))