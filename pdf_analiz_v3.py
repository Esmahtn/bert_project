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

# --- KONFİGÜRASYON ---
PDF_FILE = "2-turkiye-arnavutluk.doc" # Lütfen burayı kendi PDF adınızla değiştirin!
NER_MODEL_NAME = "savasy/bert-base-turkish-ner-cased" 
# --------------------

# Kendi Regex Kurallarınız (Kullanıcı tarafından sağlanan MASK_RULES listesi)
MASK_RULES = [

    # ------------------------------------------------------------
    # ŞİRKET / KURUM ADI
    # NOT: Bu kural, NER modeli ORG etiketini kaçırırsa devreye girer.
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
    # NOT: Bu kural, NER modeli PER etiketini kaçırırsa devreye girer.
    # ------------------------------------------------------------
    {
        "name": "kisi_adi",
        "pattern": re.compile(
            r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+[\s]+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\b"
        ),
        "replacement": "[KİŞİ_ADI]",
    },

    # ------------------------------------------------------------
    # TARAF ADI (NER'in yakalayamayacağı özelleşmiş kodlar)
    # ------------------------------------------------------------
    {
        "name": "taraf_adi",
        "pattern": re.compile(r"\bTaraf\s+[A-Z0-9]+\b", re.IGNORECASE),
        "replacement": "[TARAF_ADI]",
    },

    # ------------------------------------------------------------
    # ŞEHİR ADI (Basit)
    # NOT: NER modeli LOC etiketini kaçırırsa devreye girer.
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
    # ADRES BİLGİSİ (NER'in zorlandığı uzun ve yapısal bilgiler)
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
    # TARİH (Sayısal Format, NER'den bağımsız olarak güçlü olmalı)
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
    # SÜRE (NER tarafından yakalanması zor, sayısal/yapısal veri)
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
    # PARA / TUTAR (Sayısal/Yapısal veri)
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
    # BANKA / KİMLİK / İLETİŞİM BİLGİSİ (Hassas sayısal veriler)
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
    # Diğer etiketler (MISC/DATE) sadece çok gerekirse eklenebilir.
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
    
    # Maskeleme yapılırken indekslerin kaymaması için listeye çevirilir.
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
        # NOTE: re.sub ikinci parametrede bir string yerine bir fonksiyon bekleyebilir
        # ancak sizin replacementlarınız basit string olduğu için sub kullanıldı.
        # Adres/Vergi No gibi grupları koruyan Regex'ler için r"\1[MASKE]" formatını kullanırız.
        text = rule["pattern"].sub(rule["replacement"], text)
    return text

def pdf_metin_cikar(pdf_path):
    """PDF dosyasından tüm metni çıkarır."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
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
        results = on_isleme_ve_maskeleme(PDF_FILE)
        
        # Sadece KULLANIM ÖRNEĞİ metnini maskeleyip gösterme (Test amaçlı)
        print("\n--- TEST: KULLANICININ VERDİĞİ ÖRNEK METİN ÜZERİNDE MASKELENMİŞ ÇIKTI ---")
        
        sample_text = """
        İşbu Taşeronluk Sözleşmesi ABC Lojistik A.Ş. ile Taraf A arasında
        01.01.2023 tarihinde İstanbul'da imzalanmıştır.
        Adres: Cumhuriyet Mah. 15. Sok. No: 12 Bursa
        Vergi No: 1234567890, T.C. No: 12345678900
        Telefon: +90 532 000 00 00, E-posta: test@example.com
        IBAN: TR12 0001 2009 1234 0000 0012 34
        Proj. No: 2023-001, PO-12874
        6 ay süreyle geçerlidir. Madde 5’e göre fesih mümkündür.
        Genel Müdür Ahmet Yılmaz imzalayacaktır.
        Ek-1 Teslimat Planı ekte sunulmuştur.
        Türk Borçlar Kanunu hükümleri uygulanır. Av. Canan Çelik.
        """
        
        ner_pipeline_test = load_ner_pipeline(NER_MODEL_NAME)
        
        # Cümlelere ayırmadan tüm metni maskele
        ner_masked_test = ner_maskeleme_islemi(sample_text, ner_pipeline_test)
        final_masked_test = regex_maskeleme_islemi(ner_masked_test)
        
        print("\nMASKELENMİŞ METİN:")
        print(final_masked_test)
        
        print("\n--- PDF ÇIKTI ÖRNEKLERİ ---")
        # İlk 3 cümlenin orijinal ve maskelenmiş halini göster
        for item in results[:3]:
            print(f"\n[Cümle {item['id']}]")
            print(f"Orijinal: {item['orjinal_cumle']}")
            print(f"Maskeli:  {item['maskelenmis_cumle']}")

    except FileNotFoundError:
        print(f"\n🚨 HATA: {PDF_FILE} dosyası bulunamadı. Lütfen dosyanın adını ve yolunu kontrol edin.")
    except Exception as e:
        print(f"\n🚨 GENEL HATA OLUŞTU: {e}")