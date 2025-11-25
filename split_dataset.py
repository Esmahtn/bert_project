import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Dosyayı oku (Dosya adının 'birlesmis.csv' olduğunu varsayıyorum)
df = pd.read_csv('birlesmis_labeled.csv')

# 2. Stratified Split işlemi (%80 Train, %20 Test)
# stratify=df['label'] parametresi, sınıfların oranını korumasını sağlar.
train_df, test_df = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df['label'] # Sütun adın 'label' değilse burayı düzeltmelisin
)

# 3. Dosyaları kaydet
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

# --- KONTROL RAPORU ---
print("--- AYIRMA İŞLEMİ TAMAMLANDI ---")
print(f"Toplam Veri: {len(df)}")
print(f"Train Seti (Augmentation yapılacak): {len(train_df)} satır")
print(f"Test Seti (Dokunulmayacak): {len(test_df)} satır")

print("\n--- TRAIN SETİ DAĞILIMI ---")
print(train_df['label'].value_counts())

print("\n--- TEST SETİ DAĞILIMI ---")
print(test_df['label'].value_counts())