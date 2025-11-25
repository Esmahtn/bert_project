import re
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import torch
import nltk

# NLTK Türkçe cümle ayırıcı verisini kontrol et
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# -------------------------------------------------------------------
# --- KONFİGÜRASYON ---
# -------------------------------------------------------------------
PDF_FILE = "2-turkiye-arnavutluk.pdf" # İşlenecek PDF dosyası
NER_MODEL_NAME = "savasy/bert-base-turkish-ner-cased" 
OUTPUT_FILE = "masked_document_output.txt" # Çıktıların kaydedileceği dosya
# --------------------

# Kendi Regex Kurallarınız (MASK_RULES)
# Bu liste, önceki ile aynıdır.
MASK_RULES = [
    # ------------------------------------------------------------
    # ŞİRKET / KURUM ADI
    # ------------------------------------------------------------
    {
        "name": "sirket_adi",
        "pattern": re.compile(
            r"\b[ A-ZÇĞİÖŞÜ0-9&\.\-]+"
            r"(A\.?Ş\.?|AŞ|LTD\.?\s*ŞTİ\.?|Limited\s+Şirketi?|Anonim\s+Şirketi?)\b",
            re.IGNORECASE,
        ),
        "replacement": "[ŞİRKET_ADI]",
    },

    # ------------------------------------------------------------
    # KİŞİ ADI (Basit)
    # ------------------------------------------------------------
    {
        "name": "kisi_adi",
        "pattern": re.compile(
            r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+[\s]+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b"
        ),
        "replacement": "[KİŞİ_ADI]",
    },

    # ------------------------------------------------------------
    # TARAF ADI
    # ------------------------------------------------------------
    {
        "name": "taraf_adi",
        "pattern": re.compile(r"\bTaraf\s+[A-Z0-9]+\b", re.IGNORECASE),
        "replacement": "[TARAF_ADI]",
    },

    # ------------------------------------------------------------
    # ŞEHİR ADI (Basit)
    # ------------------------------------------------------------
    {
        "name": "yer_adi",
        "pattern": re.compile(
            r"\b(İstanbul|Ankara|İzmir|Bursa|Antalya|Adana|Konya)\b",
            re.IGNORECASE,
        ),
        "replacement": "[YER_ADI]",
    },

    # ------------------------------------------------------------
    # ADRES BİLGİSİ
    # ------------------------------------------------------------
    {
        "name": "adres_satiri",
        "pattern": re.compile(
            r"([A-ZÇĞİÖŞÜa-zçğıöşü0-9\s\.,/-]{0,80}"
            r"(Mah\.?|Mahallesi|Cad\.?|Caddesi|Sok\.?|Sokağı|Bulvarı|Blv\.?)"
            r"[A-ZÇĞİÖŞÜa-zçığıöşü0-9\s\.,/-]{0,120})"
        ),
        "replacement": "[ADRES]",
    },

    {
        "name": "adres_no",
        "pattern": re.compile(r"No[:\.]?\s*\d+\b", re.IGNORECASE),
        "replacement": "[ADRES]",
    },

    # ------------------------------------------------------------
    # TARİH (Sayısal Format)
    # ------------------------------------------------------------
    {
        "name": "tarih_sayisal",
        "pattern": re.compile(
            r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
        ),
        "replacement": "[TARİH]",
    },

    # ------------------------------------------------------------
    # TARİH (Yazılı Format)
    # ------------------------------------------------------------
    {
        "name": "tarih_yazili",
        "pattern": re.compile(
            r"\b\d{1,2}\s+"
            r"(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|"
            r"Eylül|Ekim|Kasım|Aralık)"
            r"\s+\d{2,4}\b",
            re.IGNORECASE,
        ),
        "replacement": "[TARİH]",
    },

    # ------------------------------------------------------------
    # SÜRE
    # ------------------------------------------------------------
    {
        "name": "sure",
        "pattern": re.compile(r"\b\d+\s+(gün|hafta|ay|yıl|yil)\b", re.IGNORECASE),
        "replacement": "[SÜRE]",
    },

    # ------------------------------------------------------------
    # MADDE NUMARASI / ATIFI
    # ------------------------------------------------------------
    {
        "name": "madde_no",
        "pattern": re.compile(r"\bMadde\s+\d+[a-zA-Z]?\b", re.IGNORECASE),
        "replacement": "[MADDE_NO]",
    },
    {
        "name": "madde_atifi",
        "pattern": re.compile(
            r"\bMadde\s+\d+[a-zA-Z]?['’]?(e|ye|ye göre|e göre|uyarınca)\b",
            re.IGNORECASE,
        ),
        "replacement": "[MADDE_ATIFI]",
    },

    # ------------------------------------------------------------
    # PARA / TUTAR
    # ------------------------------------------------------------
    {
        "name": "tutar",
        "pattern": re.compile(
            r"\b[\d\.]+(?:,\d+)?\s?(TL|₺|TRY|USD|EUR|Euro|Dolar)\b",
            re.IGNORECASE,
        ),
        "replacement": "[TUTAR]",
    },
    {
        "name": "para_birimi_tanim",
        "pattern": re.compile(
            r"\b(Türk\s+Lirası|Amerikan\s+Doları|Euro)\b", re.IGNORECASE
        ),
        "replacement": "[PARA_BIRIMI]",
    },

    # ------------------------------------------------------------
    # BANKA / KİMLİK / İLETİŞİM BİLGİSİ
    # ------------------------------------------------------------
    {
        "name": "iban",
        "pattern": re.compile(
            r"\bTR[0-9]{2}(?:\s?[0-9]{4}){5}\b", re.IGNORECASE
        ),
        "replacement": "[BANKA_BILGISI]",
    },
    {
        "name": "hesap_no",
        "pattern": re.compile(
            r"(Hesap\s*No[:\.]?\s*)([0-9]{6,})", re.IGNORECASE
        ),
        "replacement": r"\1[BANKA_BILGISI]",
    },
    {
        "name": "tc_kimlik",
        "pattern": re.compile(r"\b\d{11}\b"),
        "replacement": "[KIMLIK_NO]",
    },
    {
        "name": "vergi_no",
        "pattern": re.compile(
            r"(Vergi\s*No[:\.]?\s*)([0-9]{6,})", re.IGNORECASE
        ),
        "replacement": r"\1[KIMLIK_NO]",
    },
    {
        "name": "telefon",
        "pattern": re.compile(
            r"\b(?:\+?90|0)\s?(?:\(?\d{3}\)?)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"
        ),
        "replacement": "[ILETISIM_BILGISI]",
    },
    {
        "name": "email",
        "pattern": re.compile(
            r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
        "replacement": "[ILETISIM_BILGISI]",
    },

    # ------------------------------------------------------------
    # DİĞER YAPISAL VERİLER
    # ------------------------------------------------------------
    {
        "name": "url",
        "pattern": re.compile(
            r"\b(?:https?://|www\.)[^\s,]+", re.IGNORECASE
        ),
        "replacement": "[WEB_ADRESI]",
    },
    {
        "name": "vergi_dairesi",
        "pattern": re.compile(
            r"\b[A-ZÇĞİÖŞÜa-zçğıöşü\s]+Vergi\s+Dairesi\b",
            re.IGNORECASE,
        ),
        "replacement": "[VERGI_DAIRESI]",
    },
    {
        "name": "ek_referansi",
        "pattern": re.compile(
            r"\bEk[-\s]?\d+[^\n,]*", re.IGNORECASE
        ),
        "replacement": "[EK_REFERANSI]",
    },
    {
        "name": "kanun_adi",
        "pattern": re.compile(
            r"\b[ A-ZÇĞİÖŞÜa-zçğıöşü0-9\-]+"
            r"(Kanunu|Kanun|Yönetmeliği|Yönetmelik|Tebliği|Teblig)\b",
            re.IGNORECASE,
        ),
        "replacement": "[KANUN_ADI]",
    },
    {
        "name": "unvan",
        "pattern": re.compile(
            r"\b(Genel\s+Müdür|Müdür|Yönetim\s+Kurulu\s+Başkanı|"
            r"Şube\s+Müdürü|Av\.?|Dr\.?|Prof\.?\s*Dr\.?)\b",
            re.IGNORECASE,
        ),
        "replacement": "[UNVAN]",
    },
]

# NER Etiketleri ve Maske Karşılıkları
NER_MAPPING = {
    "PER": "[KİŞİ_ADI]",      # Person (Kişi Adı)
    "LOC": "[YER_ADI]",       # Location (Yer Adı)
    "ORG": "[ŞİRKET_ADI]",    # Organization (Kurum Adı)
}

def load_ner_pipeline(model_name):
    """Önceden eğitilmiş NER modelini yükler."""
    print(f"[{model_name}] NER Modeli yükleniyor...")
    return pipeline(
        "ner",
        model=model_name,
        tokenizer=model_name,
        aggregation_strategy="simple", 
        device=0 if torch.cuda.is_available() else -1
    )

def ner_maskeleme_islemi(text, ner_pipeline):
    """NER modelini kullanarak metni maskeler."""
    results = ner_pipeline(text)
    
    masked_text = list(text)
    
    for entity in reversed(results):
        label = entity['entity_group']
        start = entity['start']
        end = entity['end']
        
        if label in NER_MAPPING:
            mask = NER_MAPPING[label]
            
            # Maskeyi yerleştirme ve aradaki farkı boşlukla doldurma
            masked_text[start:end] = list(mask)
            
            if len(mask) < (end - start):
                masked_text[start + len(mask):end] = [' '] * (end - start - len(mask))
            
    return "".join(masked_text).strip()

def regex_maskeleme_islemi(text, rules=MASK_RULES):
    """Sizin Regex kurallarınızı kullanarak maskeler."""
    for rule in rules:
        text = rule["pattern"].sub(rule["replacement"], text)
    return text

def pdf_metin_cikar(pdf_path):
    """PDF dosyasından tüm metni çıkarır."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    # Çoklu boşluk ve yeni satırları temizleme
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def on_isleme_ve_maskeleme(pdf_path):
    """PDF'ten metin çıkarır, cümlelere böler ve maskeler."""
    
    # 1. Metin Çıkarma
    print(f"\n[AŞAMA 1] [{pdf_path}] dosyasından metin çıkarılıyor...")
    raw_text = pdf_metin_cikar(pdf_path)
    
    # 2. Cümlelere Ayırma
    sentences = sent_tokenize(raw_text, language='turkish')
    print(f"[AŞAMA 2] Toplam {len(sentences)} cümle bulundu.")

    # 3. NER Modelini Yükleme
    ner_pipeline = load_ner_pipeline(NER_MODEL_NAME)
    
    print("\n[AŞAMA 3] Maskeleme İşlemi Başladı (NER -> Regex)")
    
    masked_sentences = []
    
    # Her cümleyi maskeleme
    for i, sentence in enumerate(sentences):
        # Önce: NER Maskeleme (Kişi, Şirket, Yer Adları)
        ner_masked_sentence = ner_maskeleme_islemi(sentence, ner_pipeline)
        
        # Sonra: Regex Maskeleme (Tarih, Tutar, Madde No vb. yapısal veriler)
        final_masked_sentence = regex_maskeleme_islemi(ner_masked_sentence)
        
        masked_sentences.append({
            "id": i + 1,
            "orjinal_cumle": sentence,
            "maskelenmis_cumle": final_masked_sentence
        })

    return masked_sentences

if __name__ == "__main__":
    try:
        # 1. PDF dosyasından metni al, cümlelere ayır ve maskele.
        results = on_isleme_ve_maskeleme(PDF_FILE)
        
        # -------------------------------------------------------------------
        # 4. AŞAMA: Dosyaya Kayıt
        # -------------------------------------------------------------------

        print(f"\n[KAYIT AŞAMASI] {len(results)} cümle '{OUTPUT_FILE}' dosyasına kaydediliyor...")
        
        # 2. PDF Çıktılarını Dosyaya Kaydetme
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for item in results:
                # Dosyada okunması kolay, temiz bir format kullanıldı
                f.write(f"ID: {item['id']}\n")
                f.write(f"Orijinal: {item['orjinal_cumle']}\n")
                f.write(f"Maskeli:  {item['maskelenmis_cumle']}\n")
                f.write("-" * 50 + "\n")

        print(f"\n✅ KAYIT BAŞARILI! Tüm maskelenmiş çıktılar '{OUTPUT_FILE}' dosyasında.")
        
        # Opsiyonel: Kontrol için ilk maskelenmiş cümleyi terminalde gösterelim
        if len(results) > 0:
            print(f"\n--- İLK CÜMLE ÖRNEĞİ (PDF Çıktısı) ---")
            print(f"Maskeli: {results[0]['maskelenmis_cumle']}")
            print("---------------------------------------")
            
    except FileNotFoundError:
        print(f"\n🚨 HATA: {PDF_FILE} dosyası bulunamadı. Lütfen dosyanın adını ve yolunu kontrol edin.")
    except Exception as e:
        print(f"\n🚨 GENEL HATA OLUŞTU: {e}")