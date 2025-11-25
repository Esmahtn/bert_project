import pandas as pd
import os

def etiketleri_analiz_et():
    # Buraya analiz etmek istediğin dosyanın adını yaz
    dosya_adi = 'final_train_dataset.csv' 
    
    # Dosya var mı kontrol et
    if not os.path.exists(dosya_adi):
        print(f"HATA: '{dosya_adi}' bulunamadı. Lütfen dosya adını kontrol et.")
        return

    try:
        df = pd.read_csv(dosya_adi)
        
        # 'label' sütunu var mı kontrol et
        if 'label' not in df.columns:
            print("HATA: CSV dosyasında 'label' isimli bir sütun bulunamadı.")
            print(f"Mevcut sütunlar: {list(df.columns)}")
            # Eğer sütun adı farklıysa (örn: 'Label', 'etiket'), kodu ona göre düzeltmek gerekebilir.
            return

        # 1. Sayma İşlemi (Adet)
        sayimlar = df['label'].value_counts()
        
        # 2. Yüzdelik Hesaplama (Oran)
        yuzdeler = df['label'].value_counts(normalize=True) * 100

        print("\n" + "="*40)
        print(f" VERİ SETİ DAĞILIMI ({dosya_adi})")
        print("="*40)
        print(f"{'ETİKET':<20} | {'ADET':<8} | {'ORAN (%)'}")
        print("-" * 40)

        for etiket, adet in sayimlar.items():
            oran = yuzdeler[etiket]
            print(f"{str(etiket):<20} | {adet:<8} | %{oran:.2f}")
            
        print("-" * 40)
        print(f"TOPLAM SATIR: {len(df)}")
        print("="*40)

        # Ekstra Bilgi: Dengesizlik Uyarısı
        en_cok = sayimlar.iloc[0]
        en_az = sayimlar.iloc[-1]
        if en_cok / en_az > 2:
            print("\n⚠️ DİKKAT: Veri setinde ciddi dengesizlik (imbalance) var.")
            print("Modelin sadece çoğunluktaki sınıfı öğrenmeye meyilli olabilir.")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    etiketleri_analiz_et()