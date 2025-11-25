import pandas as pd

# 1. Train dosyasını oku
df = pd.read_csv('train_dataset.csv')

# 2. Etiketlere göre filtrele ve ayrı dosyalara kaydet
# Label 0 (Yüksek Risk)
df_0 = df[df['label'] == 0]
df_0.to_csv('train_label_0.csv', index=False)
print(f"Label 0 ayrıldı: {len(df_0)} satır -> 'train_label_0.csv'")

# Label 1 (Orta Risk)
df_1 = df[df['label'] == 1]
df_1.to_csv('train_label_1.csv', index=False)
print(f"Label 1 ayrıldı: {len(df_1)} satır -> 'train_label_1.csv'")

# Label 2 (Düşük Risk)
df_2 = df[df['label'] == 2]
df_2.to_csv('train_label_2.csv', index=False)
print(f"Label 2 ayrıldı: {len(df_2)} satır -> 'train_label_2.csv'")