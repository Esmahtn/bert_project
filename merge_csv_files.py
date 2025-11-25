# Bu kod csv dosyalarını tek bir csv dosyasına dönüştürmek için
import pandas as pd
import glob
import os

def csvleri_duzgun_birlestir():
    klasor_yolu = 'labeled_dataset'
    dosya_yollari = glob.glob(os.path.join(klasor_yolu, "*.csv"))
    
    if not dosya_yollari:
        print("Dosya bulunamadı.")
        return

    dataframeler = []

    print("İşlem başlıyor...")

    for dosya in dosya_yollari:
        try:
            # 1. Dosyayı oku
            # header=0 -> İlk satırın başlık olduğunu varsayar.
            df = pd.read_csv(dosya, header=0)
            
            # 2. GÜVENLİK KONTROLÜ: Sütun sayısını kontrol et
            # Eğer dosya 2 sütunlu değilse (boş veya hatalıysa) uyar ve atla
            if len(df.columns) != 2:
                print(f"DİKKAT: '{os.path.basename(dosya)}' dosyası 2 sütunlu değil, atlanıyor.")
                continue

            # 3. KRİTİK DÜZELTME: Sütun isimlerini zorla değiştiriyoruz.
            # Dosyada başlık ne yazarsa yazsın (Review, Label, Text vb.)
            # biz hepsini 'text' ve 'label' olarak etiketliyoruz.
            df.columns = ['text', 'label']
            
            dataframeler.append(df)
            
        except Exception as e:
            print(f"HATA: {dosya} okunurken hata oluştu: {e}")

    if dataframeler:
        # 4. Birleştirme
        birlestirilmis_df = pd.concat(dataframeler, ignore_index=True)
        
        # 5. Varsa tamamen boş satırları temizle
        birlestirilmis_df.dropna(how='all', inplace=True)

        # Kaydet
        cikis_dosyasi = 'duzeltilmis_veri_seti.csv'
        birlestirilmis_df.to_csv(cikis_dosyasi, index=False)
        
        print("-" * 30)
        print("SORUN ÇÖZÜLDÜ.")
        print(f"Dosya Adı: {cikis_dosyasi}")
        print(f"Toplam Satır: {len(birlestirilmis_df)}")
        print(f"Sütunlar: {list(birlestirilmis_df.columns)}")
    else:
        print("Birleştirilecek uygun veri bulunamadı.")

if __name__ == "__main__":
    csvleri_duzgun_birlestir()