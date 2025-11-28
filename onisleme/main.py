import pandas as pd
import os
# Kendi yazdığımız modülü buradan çağırıyoruz
from sentence_splitter import metni_cumlelere_ayir

def main():
    # --- AYARLAR ---
    input_pdf = "/Users/pelinsusaglam/Desktop/metinonisleme/sozlesme.pdf"  # İşlenecek dosya adı (senin dosyanın adı neyse onu yaz)
    output_csv = "islenmis_veriler.csv"
    # ---------------

    print("-" * 40)
    print(f"🚀 Süreç Başlıyor: {input_pdf}")

    # 1. ADIM: PDF İşleme ve Cümle Ayırma
    print("1️⃣  PDF okunuyor ve cümlelere ayrılıyor...")
    cumle_listesi = metni_cumlelere_ayir(input_pdf)

    if not cumle_listesi:
        print("❌ İşlem başarısız veya metin bulunamadı.")
        return

    print(f"✅ Toplam {len(cumle_listesi)} cümle ayrıştırıldı.")

    # 2. ADIM: (Buraya İleride BERT Gelecek)
    # Şimdilik sadece CSV'ye kaydediyoruz.
    print("2️⃣  Veriler kaydediliyor...")
    
    df = pd.DataFrame(cumle_listesi, columns=["Cumle"])
    
    # utf-8-sig: Excel'de Türkçe karakterlerin bozuk çıkmaması için önemli
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"💾 Dosya kaydedildi: {os.path.abspath(output_csv)}")
    print("-" * 40)
    print("🎉 Görev Tamamlandı.")

if __name__ == "__main__":
    main()