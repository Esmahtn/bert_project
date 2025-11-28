import fitz  # PyMuPDF
import spacy
import os
import re

# --- MODEL YÜKLEME ---
print("⏳ SpaCy Türkçe Modeli (Large) yükleniyor...")
model_adi = "tr_core_news_lg"

try:
    nlp = spacy.load(model_adi)
    print(f"✅ Model hazır: {model_adi}")
except OSError:
    print(f"❌ HATA: '{model_adi}' modeli bulunamadı.")
    exit()

def cop_cumleleri_temizle(cumle_listesi):
    """
    Spacy çıktısındaki gürültüleri (sayfa no, başlık artığı, boşluk) temizler.
    """
    temiz_liste = []
    
    for cumle in cumle_listesi:
        # 1. Satır başı/sonu boşluklarını temizle
        c = cumle.strip()
        
        # 2. BOŞ VEYA ÇOK KISA SATIRLARI AT
        # Örn: ".", "A", "1." gibi 4 karakterden kısa şeyler genellikle çöp veridir.
        if len(c) < 4:
            continue
            
        # 3. SADECE SAYI OLANLARI AT (Sayfa numaraları)
        # Örn: "12", "4", "1995"
        if c.isdigit():
            continue
            
        # 4. SADECE NOKTALAMA VE SAYI İÇERENLERİ AT
        # Örn: "1.2.", "---", "..."
        # Harf içermeyen satırları temizle
        if not re.search('[a-zA-ZğüşıöçĞÜŞİÖÇ]', c):
            continue

        # 5. GEÇERSİZ BAŞLANGIÇLARI TEMİZLE (İsteğe bağlı)
        # Bazen PDF'ten "......" veya "__________" gibi çizgiler gelir.
        # Eğer satırın yarısı noktaysa at.
        if c.count(".") > len(c) / 2:
            continue

        temiz_liste.append(c)
        
    return temiz_liste

def metni_cumlelere_ayir(pdf_yolu):
    """
    PDF dosyasını okur, temizler ve filtrelerden geçirip cümle listesi döndürür.
    """
    if not os.path.exists(pdf_yolu):
        print(f"⚠️ Dosya bulunamadı: {pdf_yolu}")
        return []

    # 1. PDF Okuma
    try:
        doc = fitz.open(pdf_yolu)
    except Exception as e:
        print(f"⚠️ PDF okuma hatası: {e}")
        return []

    full_text = ""
    for page in doc:
        # Header ve Footer'ları (Sayfa altı/üstü) hariç tutmak için 
        # basit bir kırpma (crop) mantığı eklenebilir ama şimdilik metni alıyoruz.
        full_text += page.get_text("text") + " "

    # 2. Ham Metin Temizliği
    # Fazla boşlukları ve satır atlamalarını tek boşluğa indir
    clean_text = " ".join(full_text.split())

    # 3. SpaCy ile Analiz
    nlp.max_length = 3000000 
    doc_spacy = nlp(clean_text)

    # 4. Cümleleri Çıkar
    ham_cumleler = [sent.text.strip() for sent in doc_spacy.sents]
    
    # 5. FİLTRELEME İŞLEMİ (Yeni Eklenen Kısım)
    final_cumleler = cop_cumleleri_temizle(ham_cumleler)

    return final_cumleler
