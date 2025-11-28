import matplotlib.pyplot as plt

# Akış adımları (kutulardaki yazılar)
steps = [
    "4 model ile ön kıyaslama\n(Epoch=5, Batch=16, LR=3e-5)",
    "En iyi iki modelin seçilmesi\n(Legal-BERT & Law-EQA)",
    "5 epoch sabit tutularak\nilk LR odaklı optimizasyon\n(çeşitli öğrenme oranları,\nperformans tavanı ≈ %86.5)",
    "Epoch=8 ile odaklanmış\nhiperparametre optimizasyonu\n(Batch ve LR varyasyonları)",
    "Nihai modelin seçimi\n(Law-EQA V3, F1=0.9202)",
    "Kalitatif analiz ve\nhata tipi incelemesi"
]

# Şekil ve eksen ayarları
fig, ax = plt.subplots(figsize=(14, 4))  # yükseklik biraz arttı
ax.set_xlim(0, len(steps) * 2)
ax.set_ylim(0, 2)
ax.axis('off')  # eksenleri gizle

# Kutu boyutları ve konumlar
box_width = 2.0
box_height = 1.0
y = 1.0  # tüm kutular aynı yatay hizada

# x merkezleri: 1, 3, 5, 7, 9, 11 gibi
x_positions = [1 + 2 * i for i in range(len(steps))]

# Kutuları ve metinleri çiz
for x, text in zip(x_positions, steps):
    # Dikdörtgen (kutu)
    rect = plt.Rectangle(
        (x - box_width / 2, y - box_height / 2),
        box_width,
        box_height,
        fill=False  # içi boş, sadece çerçeve
    )
    ax.add_patch(rect)
    # Kutu içi metin
    ax.text(
        x, y, text,
        ha='center', va='center',
        fontsize=10,  # BURAYI BÜYÜTTÜK
        wrap=True
    )

# Kutular arasındaki oklar
for i in range(len(x_positions) - 1):
    x1 = x_positions[i] + box_width / 2   # sağ kenar
    x2 = x_positions[i + 1] - box_width / 2  # sonraki kutunun sol kenarı
    ax.annotate(
        "",
        xy=(x2, y),
        xytext=(x1, y),
        arrowprops=dict(arrowstyle="->")
    )

# Başlık
plt.title(
    "Şekil 4.3. Sözleşme maddesi risk sınıflandırma görevinde "
    "model seçim sürecinin akış diyagramı",
    fontsize=12  # BAŞLIK BOYUTU DA ARTTI
)

plt.tight_layout()
plt.savefig("sekil_4_3_model_secim_sureci_akisi_guncel.png", dpi=300, bbox_inches="tight")
plt.show()
