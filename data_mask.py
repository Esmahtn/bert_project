import re

# -------------------------------------------------------------------
# HER KATEGORİ İÇİN REGEX + MASKE KURALLARI
# Her regex'in üstünde ÖRNEK kullanım açıklaması vardır.
# -------------------------------------------------------------------
MASK_RULES = [

    # ------------------------------------------------------------
    # ŞİRKET / KURUM ADI
    # Örnek:
    #   "ABC Lojistik A.Ş."
    #   "XYZ LTD. ŞTİ."
    #   "Delta Anonim Şirketi"
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
    # KİŞİ ADI (basit 2 kelime, ikisi de büyük harfle başlıyor)
    # Örnek:
    #   "Ahmet Yılmaz"
    #   "Mehmet Ali"
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
    # Örnek:
    #   "Taraf A"
    #   "Taraf B"
    #   "Taraf 1"
    # ------------------------------------------------------------
    {
        "name": "taraf_adi",
        "pattern": re.compile(r"\bTaraf\s+[A-Z0-9]+\b", re.IGNORECASE),
        "replacement": "[TARAF_ADI]",
    },

    # ------------------------------------------------------------
    # ŞEHİR ADI (örnek bazı şehirler)
    # Örnek:
    #   "İstanbul"
    #   "Ankara"
    #   "Bursa"
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
    # Örnek:
    #   "Cumhuriyet Mah. 12. Sok. No:3"
    #   "Atatürk Caddesi No:45"
    #   "Barış Mahallesi 8. Sok."
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

    # "No: 3"
    {
        "name": "adres_no",
        "pattern": re.compile(r"No[:\.]?\s*\d+\b", re.IGNORECASE),
        "replacement": "[ADRES]",
    },

    # ------------------------------------------------------------
    # TARİH (sayısal format)
    # Örnek:
    #   "01.01.2023"
    #   "1/1/2022"
    #   "2023-01-05"
    # ------------------------------------------------------------
    {
        "name": "tarih_sayisal",
        "pattern": re.compile(
            r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b"
        ),
        "replacement": "[TARİH]",
    },

    # ------------------------------------------------------------
    # TARİH (yazılı format)
    # Örnek:
    #   "5 Ocak 2023"
    #   "12 Aralık 2022"
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
    # Örnek:
    #   "6 ay"
    #   "2 yıl"
    #   "15 gün"
    # ------------------------------------------------------------
    {
        "name": "sure",
        "pattern": re.compile(r"\b\d+\s+(gün|hafta|ay|yıl|yil)\b", re.IGNORECASE),
        "replacement": "[SÜRE]",
    },

    # ------------------------------------------------------------
    # MADDE NUMARASI
    # Örnek:
    #   "Madde 5"
    #   "Madde 10/a"
    # ------------------------------------------------------------
    {
        "name": "madde_no",
        "pattern": re.compile(r"\bMadde\s+\d+[a-zA-Z]?\b", re.IGNORECASE),
        "replacement": "[MADDE_NO]",
    },

    # "Madde 5’e göre"
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
    # Örnek:
    #   "10.000 TL"
    #   "5,500 USD"
    #   "200 EUR"
    # ------------------------------------------------------------
    {
        "name": "tutar",
        "pattern": re.compile(
            r"\b[\d\.]+(?:,\d+)?\s?(TL|₺|TRY|USD|EUR|Euro|Dolar)\b",
            re.IGNORECASE,
        ),
        "replacement": "[TUTAR]",
    },

    # "Türk Lirası"
    {
        "name": "para_birimi_tanim",
        "pattern": re.compile(
            r"\b(Türk\s+Lirası|Amerikan\s+Doları|Euro)\b", re.IGNORECASE
        ),
        "replacement": "[PARA_BIRIMI]",
    },

    # ------------------------------------------------------------
    # BANKA BİLGİSİ
    # Örnek:
    #   "TR12 0001 2009 1234 0000 0012 34" (IBAN)
    #   "Hesap No: 1234567890"
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

    # ------------------------------------------------------------
    # KİMLİK / VERGİ NO
    # Örnek:
    #   "12345678900" (T.C. Kimlik)
    #   "Vergi No: 1234567890"
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # TELEFON / E-POSTA / FAKS
    # Örnek:
    #   "+90 532 000 00 00"
    #   "test@example.com"
    # ------------------------------------------------------------
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
    # WEB ADRESİ / URL
    # Örnek:
    #   "https://abc.com"
    #   "www.example.org"
    # ------------------------------------------------------------
    {
        "name": "url",
        "pattern": re.compile(
            r"\b(?:https?://|www\.)[^\s,]+", re.IGNORECASE
        ),
        "replacement": "[WEB_ADRESI]",
    },

    # ------------------------------------------------------------
    # VERGİ DAİRESİ
    # Örnek:
    #   "Bursa Vergi Dairesi"
    # ------------------------------------------------------------
    {
        "name": "vergi_dairesi",
        "pattern": re.compile(
            r"\b[A-ZÇĞİÖŞÜa-zçğıöşü\s]+Vergi\s+Dairesi\b",
            re.IGNORECASE,
        ),
        "replacement": "[VERGI_DAIRESI]",
    },

    # ------------------------------------------------------------
    # EK / DOSYA REFERANSI
    # Örnek:
    #   "Ek-1 Teslimat Planı"
    #   "Ek 2"
    # ------------------------------------------------------------
    {
        "name": "ek_referansi",
        "pattern": re.compile(
            r"\bEk[-\s]?\d+[^\n,]*", re.IGNORECASE
        ),
        "replacement": "[EK_REFERANSI]",
    },

    # ------------------------------------------------------------
    # KANUN ADI
    # Örnek:
    #   "Türk Borçlar Kanunu"
    #   "İş Sağlığı ve Güvenliği Yönetmeliği"
    # ------------------------------------------------------------
    {
        "name": "kanun_adi",
        "pattern": re.compile(
            r"\b[ A-ZÇĞİÖŞÜa-zçğıöşü0-9\-]+"
            r"(Kanunu|Kanun|Yönetmeliği|Yönetmelik|Tebliği|Teblig)\b",
            re.IGNORECASE,
        ),
        "replacement": "[KANUN_ADI]",
    },

    # ------------------------------------------------------------
    # ÜNVAN
    # Örnek:
    #   "Genel Müdür"
    #   "Av."
    #   "Dr."
    # ------------------------------------------------------------
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


# -------------------------------------------------------------------
# ANA MASKELEME FONKSİYONU
# -------------------------------------------------------------------
def mask_text(text: str) -> str:
    for rule in MASK_RULES:
        text = rule["pattern"].sub(rule["replacement"], text)
    return text


# -------------------------------------------------------------------
# KULLANIM ÖRNEĞİ
# -------------------------------------------------------------------
if __name__ == "__main__":
    sample = """
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
    """

    print("MASKELENMİŞ METİN:\n")
    print(mask_text(sample))
