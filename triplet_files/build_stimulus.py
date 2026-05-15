"""
build_stimulus.py
=================

Türkçe morfoloji-duyarlı triplet probe stimulus üretici.

Tasarım:
  - 16 frekans-eşleştirilmiş somut isim kökü (8 ince, 8 kalın)
  - 6 ek tipi (3 kategori): inflectional, light derivational, strong derivational
  - Her ek için 5 sabit A-B kök çifti
  - Vokal sınıfı içinde eşleştirilmiş triplet'ler

Üretilen dosyalar:
  - triplet_stimulus.csv : tüm triplet'ler ve metadata
  - root_inventory.csv   : kullanılan kök listesi + frekanslar
  - suffix_inventory.csv : kullanılan ek formları
"""

import csv
from pathlib import Path
from wordfreq import zipf_frequency


# =====================================================================
# 1. KÖK SEÇİMİ — frekans-eşleştirilmiş 8+8 somut isim
# =====================================================================
# Notlar:
#   - Frekansları benzer bantta (Zipf 4.5-5.5 arası mümkün olduğunca)
#   - Hepsi somut isim (concrete count noun)
#   - 6 ekin tümünü morfolojik olarak alabilir
#   - "ev" sınır üstü (5.57) ama klasik test kelimesi olarak dahil

INCE_ROOTS = [
    "ev",       # 5.57 — house
    "göz",      # 5.47 — eye
    "deniz",    # 5.39 — sea
    "şehir",    # 5.34 — city
    "köpek",    # 4.98 — dog
    "çiçek",    # 4.92 — flower
    "gemi",     # 4.91 — ship
    "ekmek",    # 4.81 — bread
]

KALIN_ROOTS = [
    "kitap",    # 5.41 — book
    "okul",     # 5.32 — school
    "kalp",     # 5.14 — heart
    "taş",      # 5.13 — stone
    "kapı",     # 5.04 — door
    "ayak",     # 4.98 — foot
    "kafa",     # 4.93 — head
    "masa",     # 4.56 — table
]


# =====================================================================
# 2. EKLER — 6 ek, 3 kategori
# =====================================================================
# Template format:
#   A = 2-way vokal (a/e)
#   I = 4-way vokal (ı/i/u/ü)
#   D = konsonant uyumu (d/t)

SUFFIXES = [
    ("inflectional", "PLURAL",   "lAr"),
    ("inflectional", "LOCATIVE", "DA"),
    ("inflectional", "ABLATIVE", "DAn"),
    ("light_deriv",  "WITH",     "lI"),
    ("light_deriv",  "NOMINAL",  "lIk"),
    ("strong_deriv", "WITHOUT",  "sIz"),
]


# =====================================================================
# 3. TÜRKÇE VOKAL & KONSONANT UYUMU
# =====================================================================

FRONT_VOWELS = set('eiöü')
BACK_VOWELS = set('aıou')
HARD_CONSONANTS = set('pçtkfhsş')  # sertleşmeye yol açanlar

# 4-way (i/ı/u/ü) son sesli harf -> ek vokali eşlemesi
FOUR_WAY_MAP = {
    'a': 'ı', 'ı': 'ı',
    'e': 'i', 'i': 'i',
    'o': 'u', 'u': 'u',
    'ö': 'ü', 'ü': 'ü',
}


def vowel_class(stem):
    """ince / kalın (son sesli harfin ön/arka sınıfı)"""
    for ch in reversed(stem.lower()):
        if ch in FRONT_VOWELS:
            return 'ince'
        if ch in BACK_VOWELS:
            return 'kalın'
    return None


def two_way_vowel(stem):
    """a/e — son sesli harfe göre"""
    vc = vowel_class(stem)
    return 'e' if vc == 'ince' else 'a'


def four_way_vowel(stem):
    """ı/i/u/ü — son sesli harfe göre"""
    for ch in reversed(stem.lower()):
        if ch in FOUR_WAY_MAP:
            return FOUR_WAY_MAP[ch]
    return 'i'


def apply_morphology(stem, suffix_template):
    """
    Template'i instantiate et:
      A -> a/e
      I -> ı/i/u/ü
      D -> d/t (konsonant uyumu)
    """
    suf = suffix_template
    suf = suf.replace('A', two_way_vowel(stem))
    suf = suf.replace('I', four_way_vowel(stem))
    
    # Konsonant uyumu: d başında ve önceki ünsüz sertse t olur
    if 'D' in suf:
        last_char = stem[-1].lower()
        d_replacement = 't' if last_char in HARD_CONSONANTS else 'd'
        suf = suf.replace('D', d_replacement)
    
    return stem + suf


# =====================================================================
# 4. TRIPLET ÜRETİMİ
# =====================================================================

def build_pairs(roots):
    """
    Her vokal sınıfı içinde 5 sabit A-B kök çifti üret.
    Stratejimiz: 8 köktü en yakın frekanslı 2'şer eşleştir.
    Bu sayede frekans eşleşmesi triplet-içi de optimal.
    """
    # Roots zaten frekans sırasında. Ardışık eşleştirme:
    # (0,1), (2,3), (4,5), (6,7) + ek olarak (0,2)
    pairs = [
        (roots[0], roots[1]),
        (roots[2], roots[3]),
        (roots[4], roots[5]),
        (roots[6], roots[7]),
        (roots[1], roots[2]),  # 5. çift: ortadan ek bir kombinasyon
    ]
    return pairs


def build_triplets(roots, vowel_label):
    """
    Her ek için: her A-B çifti için bir triplet.
    Triplet = (A+suffix1, A+suffix2, B+suffix1) -- klasik probe yapısı.
    
    Yapı:
      Her satır = bir triplet
      - Aim: model k vs ek bilgisini ne kadar weight veriyor?
      
      Karşılaştırmalar:
        sim(A+x, A+y) : aynı kök, farklı ek
        sim(A+x, B+x) : farklı kök, aynı ek
        Δ = sim(A+x, A+y) − sim(A+x, B+x)
        (Δ > 0 → kök bilgisi baskın
         Δ < 0 → ek bilgisi baskın)
    """
    pairs = build_pairs(roots)
    triplets = []
    
    suffix_labels = [s[1] for s in SUFFIXES]  # PLURAL, LOCATIVE, ...
    
    triplet_id = 0
    for cat, primary_suffix_label, primary_template in SUFFIXES:
        # Diğer eklerden birini "secondary" olarak kullan
        # Strateji: her zaman PLURAL'ı secondary tut, primary olduğunda LOCATIVE'i kullan
        if primary_suffix_label == 'PLURAL':
            secondary_template = 'DA'
            secondary_label = 'LOCATIVE'
            secondary_cat = 'inflectional'
        else:
            secondary_template = 'lAr'
            secondary_label = 'PLURAL'
            secondary_cat = 'inflectional'
        
        for A, B in pairs:
            triplet_id += 1
            A_x = apply_morphology(A, primary_template)
            A_y = apply_morphology(A, secondary_template)
            B_x = apply_morphology(B, primary_template)
            
            triplets.append({
                'triplet_id': f"T{triplet_id:03d}",
                'vowel_class': vowel_label,
                'category': cat,
                'primary_suffix': primary_suffix_label,
                'secondary_suffix': secondary_label,
                'root_A': A,
                'root_B': B,
                'A_x': A_x,           # A + primary
                'A_y': A_y,           # A + secondary  
                'B_x': B_x,           # B + primary
                'zipf_A': round(zipf_frequency(A, 'tr'), 2),
                'zipf_B': round(zipf_frequency(B, 'tr'), 2),
                'zipf_diff': round(abs(zipf_frequency(A, 'tr') - zipf_frequency(B, 'tr')), 2),
            })
    
    return triplets


# =====================================================================
# 5. ANA RUTİN — CSV üret
# =====================================================================

def main(output_dir='.'):
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    
    # 1) Root inventory
    print("Root inventory yazılıyor...")
    with open(out / 'root_inventory.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['root', 'vowel_class', 'zipf_frequency'])
        for r in INCE_ROOTS:
            w.writerow([r, 'ince', round(zipf_frequency(r, 'tr'), 2)])
        for r in KALIN_ROOTS:
            w.writerow([r, 'kalın', round(zipf_frequency(r, 'tr'), 2)])
    
    # 2) Suffix inventory (örnek bir ince ve bir kalın kök için)
    print("Suffix inventory yazılıyor...")
    with open(out / 'suffix_inventory.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['category', 'label', 'template',
                    'form_after_ev_ince', 'form_after_kitap_kalın'])
        for cat, label, tmpl in SUFFIXES:
            ev_form = apply_morphology('ev', tmpl)
            kitap_form = apply_morphology('kitap', tmpl)
            w.writerow([cat, label, tmpl, ev_form[2:], kitap_form[5:]])
    
    # 3) Triplet stimulus
    print("Triplet stimulus üretiliyor...")
    triplets = []
    triplets += build_triplets(INCE_ROOTS, 'ince')
    triplets += build_triplets(KALIN_ROOTS, 'kalın')
    
    with open(out / 'triplet_stimulus.csv', 'w', newline='', encoding='utf-8') as f:
        if triplets:
            w = csv.DictWriter(f, fieldnames=list(triplets[0].keys()))
            w.writeheader()
            w.writerows(triplets)
    
    # 4) Özet
    print(f"\n✓ Üretildi: {len(triplets)} triplet")
    print(f"  - İnce ünlü triplet'leri: {sum(1 for t in triplets if t['vowel_class']=='ince')}")
    print(f"  - Kalın ünlü triplet'leri: {sum(1 for t in triplets if t['vowel_class']=='kalın')}")
    print(f"  - Kategori başına: ", end='')
    from collections import Counter
    cat_counts = Counter(t['category'] for t in triplets)
    print(', '.join(f"{c}={n}" for c, n in cat_counts.items()))
    
    return triplets


if __name__ == '__main__':
    triplets = main(output_dir='/home/claude/triplet_probe/output')
    
    print("\n=== ÖRNEK TRIPLETLER (her kategori için 1) ===")
    seen_cats = set()
    for t in triplets:
        key = (t['vowel_class'], t['category'], t['primary_suffix'])
        if key in seen_cats:
            continue
        seen_cats.add(key)
        print(f"\n[{t['triplet_id']}] {t['vowel_class']} | {t['category']} | {t['primary_suffix']}")
        print(f"   A+x = {t['A_x']}     (A={t['root_A']}, Zipf={t['zipf_A']})")
        print(f"   A+y = {t['A_y']}     (A={t['root_A']} + {t['secondary_suffix']})")
        print(f"   B+x = {t['B_x']}     (B={t['root_B']}, Zipf={t['zipf_B']})")
        print(f"   Frekans farkı: {t['zipf_diff']}")
