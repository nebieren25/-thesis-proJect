"""
check_tokenization.py
=====================

Bu scripti KENDİ MAKİNENDE çalıştır.

Amaç:
  Her stimulus kelimesini BERTurk tokenizer'ı ile tokenize edip
  kaç sub-token'a bölündüğünü görmek. Triplet probe için her kelimenin
  EN AZ 2 sub-token'a bölünmesi tercih edilir (yoksa "subtoken pooling"
  diye bir şey yok).

Kullanım:
  python3 check_tokenization.py triplet_stimulus.csv

Çıktı:
  triplet_stimulus_with_tokenization.csv
    - Her A_x, A_y, B_x için sub-token sayısı
    - Hangi triplet'ler "iyi" (her üç kelime de split)
    - Hangi triplet'ler problemli
"""

import sys
import csv
from pathlib import Path

# Tek bağımlılık: transformers (BERTurk tokenizer için)
from transformers import AutoTokenizer


MODELS = {
    'BERTurk': 'dbmdz/bert-base-turkish-cased',
    'mBERT':   'bert-base-multilingual-cased',
    'XLM-R':   'xlm-roberta-base',
}


def main(stimulus_csv):
    print("Tokenizer'ları yüklüyorum...")
    tokenizers = {}
    for name, model_id in MODELS.items():
        print(f"  - {name}")
        tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
    
    print(f"\nStimulus dosyasını okuyorum: {stimulus_csv}")
    rows = []
    with open(stimulus_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  {len(rows)} triplet yüklendi")
    
    # Her A_x, A_y, B_x için her modelde tokenize et
    print("\nTokenize ediliyor...")
    enriched = []
    for row in rows:
        new_row = dict(row)
        for word_key in ['A_x', 'A_y', 'B_x']:
            word = row[word_key]
            for model_name, tok in tokenizers.items():
                # tokenize without special tokens
                tokens = tok.tokenize(word)
                new_row[f'{word_key}_{model_name}_ntokens'] = len(tokens)
                new_row[f'{word_key}_{model_name}_tokens'] = '|'.join(tokens)
        
        # "Good triplet" kriteri: BERTurk'te her 3 kelime de split (>=2 token)
        nt_ax = new_row['A_x_BERTurk_ntokens']
        nt_ay = new_row['A_y_BERTurk_ntokens']
        nt_bx = new_row['B_x_BERTurk_ntokens']
        all_split = (nt_ax >= 2 and nt_ay >= 2 and nt_bx >= 2)
        new_row['BERTurk_all_split'] = all_split
        
        # Eklerin görünür olduğunu kontrol: A_x'in son token'ı
        # B_x'in son token'ı ile aynı mı? (aynı ek olduğu için olmalı)
        ax_tokens = new_row[f'A_x_BERTurk_tokens'].split('|')
        bx_tokens = new_row[f'B_x_BERTurk_tokens'].split('|')
        same_final_token = ax_tokens[-1] == bx_tokens[-1] if ax_tokens and bx_tokens else False
        new_row['BERTurk_shared_final_subtoken'] = same_final_token
        
        enriched.append(new_row)
    
    # Yaz
    output_path = Path(stimulus_csv).parent / 'triplet_stimulus_with_tokenization.csv'
    print(f"\nYazıyorum: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
        writer.writeheader()
        writer.writerows(enriched)
    
    # Özet istatistikleri
    print("\n" + "=" * 60)
    print("ÖZET (BERTurk)")
    print("=" * 60)
    
    n = len(enriched)
    n_all_split = sum(1 for r in enriched if r['BERTurk_all_split'])
    n_shared = sum(1 for r in enriched if r['BERTurk_shared_final_subtoken'])
    
    print(f"Toplam triplet: {n}")
    print(f"  Her 3 kelime de split (good): {n_all_split} ({100*n_all_split/n:.0f}%)")
    print(f"  A_x ve B_x aynı son sub-token: {n_shared} ({100*n_shared/n:.0f}%)")
    print()
    print("İyi triplet sayısı kategori bazında:")
    from collections import Counter
    cat_total = Counter()
    cat_good = Counter()
    for r in enriched:
        cat_total[r['category']] += 1
        if r['BERTurk_all_split']:
            cat_good[r['category']] += 1
    for cat in cat_total:
        print(f"  {cat:<15} {cat_good[cat]}/{cat_total[cat]}")
    
    print("\nİyi triplet sayısı ek bazında:")
    suf_total = Counter()
    suf_good = Counter()
    for r in enriched:
        suf_total[r['primary_suffix']] += 1
        if r['BERTurk_all_split']:
            suf_good[r['primary_suffix']] += 1
    for suf in suf_total:
        print(f"  {suf:<15} {suf_good[suf]}/{suf_total[suf]}")
    
    # Problemli triplet'leri listele
    print("\nProblemli triplet'ler (BERTurk'te en az bir kelime split olmuyor):")
    for r in enriched:
        if not r['BERTurk_all_split']:
            print(f"  [{r['triplet_id']}] {r['A_x']}({r['A_x_BERTurk_ntokens']}) "
                  f"{r['A_y']}({r['A_y_BERTurk_ntokens']}) "
                  f"{r['B_x']}({r['B_x_BERTurk_ntokens']})")
    
    print(f"\n✓ Done. Output: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # Default path
        main('triplet_stimulus.csv')
