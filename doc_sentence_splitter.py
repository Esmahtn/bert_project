#Bu kod, seçtiğin PDF veya DOCX sözleşme dosyasını okuyup içindeki metni alıyor, sonra metni noktalama işaretlerine göre cümlelere ayırıyor, tarihleri ve kısaltmaları karıştırmamak için koruyor ve temizliyor.Son olarak tüm cümleleri bir liste hâline getirip CSV dosyası olarak kaydediyor.
import os
import re
import fitz  # PyMuPDF
import docx
import pandas as pd

# 🔹 Denemek istediğin dosya yolu
file_path = "/Users/pelinsusaglam/Desktop/dataset_duzenle/data/2-turkiye-arnavutluk.docx"  # veya .docx
output_path = "/Users/pelinsusaglam/Desktop/dataset_duzenle/test_output.csv"

def split_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()

    # Kısaltmalar ve tarihleri belirle
    abbreviations = ["Mr", "Mrs", "Dr", "Prof", "Sn", "T.C", "No", "Madde", "Md", "Bkz"]
    date_pattern = r"\d{1,2}\.\d{1,2}\.\d{2,4}"

    # Önce tarihler ve kısaltmaları geçici olarak koruyalım
    protected = {}

    # 1. Kısaltmaları koru
    for i, abbr in enumerate(abbreviations):
        text = text.replace(abbr + ".", f"__ABBR{i}__")
        protected[f"__ABBR{i}__"] = abbr + "."

    # 2. Tarihleri koru
    matches = re.findall(date_pattern, text)
    for i, m in enumerate(matches):
        text = text.replace(m, f"__DATE{i}__")
        protected[f"__DATE{i}__"] = m

    # 3. Nokta, ünlem, soru işareti ile böl
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # 4. Korunan yerleri geri getir
    restored = []
    for s in sentences:
        for key, val in protected.items():
            s = s.replace(key, val)
        s = s.strip()
        if not s:
            continue
        # “Tarih” içeren satırları atla
        if re.search(r"tarih|TARİH|\d{1,2}\.\d{1,2}\.\d{2,4}", s):
            continue
        restored.append(s)

    return restored


# --- Dosyayı oku ---
text = ""
if file_path.lower().endswith(".pdf"):
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"

elif file_path.lower().endswith(".docx"):
    doc = docx.Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs])
else:
    raise ValueError("Desteklenmeyen dosya türü. Sadece PDF veya DOCX olmalı.")


# --- İşle ve kaydet ---
if text.strip():
    sentences = split_sentences(text)
    df = pd.DataFrame(sentences, columns=["text"])
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ '{os.path.basename(file_path)}' dosyasında {len(sentences)} cümle bulundu.")
    print(f"💾 CSV olarak kaydedildi → {output_path}\n")
    print("🔹 İlk 10 cümle:")
    for s in sentences[:10]:
        print("-", s)
else:
    print("⚠️ Dosyada metin bulunamadı veya okunamadı.")