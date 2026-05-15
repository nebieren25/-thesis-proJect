# Morphological Triplet Probe — Implementation Plan

> **Status:** Stimulus designed. Ready for tokenization check and embedding pipeline implementation.
> **Owner:** Nebi Eren Aygün
> **Supervisor:** Dr. Giovanni Cassani
> **Last updated:** 15 May 2026
> **Target deadline:** First submission 22 May 2026

---

## 1. Background & Motivation

### 1.1 What is this probe?

This is a **follow-up experiment** to the main AnlamVer-based subword pooling study. It does not replace the main analysis; it complements it.

The main experiment shows **which pooling strategy** (first, last, mean, max) best matches human similarity judgments. Current best result: **BERTurk, layer 7, first pooling, Spearman ρ ≈ 0.5275** (contextual condition).

But the main experiment does **not answer**: *why* does first pooling win? Is it because the first sub-token carries stem (kök) information, while later sub-tokens carry suffix (ek) information? This probe directly tests that question.

### 1.2 Research question

> **Does the model encode word meaning primarily through the stem (kök), through the suffix (ek), or through some combination?**

The supervisor explicitly framed this as the appropriate question in Meeting 6: *"You have the hypothesis that meaning might be more or less scattered across tokens."*

### 1.3 Probe design — triplet structure

Each stimulus is a triplet of three Turkish words:

```
A+x   (root A + suffix x)        e.g.  evler   = ev + lAr
A+y   (root A + suffix y)        e.g.  evde    = ev + DA
B+x   (root B + suffix x)        e.g.  gözler  = göz + lAr
```

Comparisons:

- `sim(A+x, A+y)` → same root, different suffix
- `sim(A+x, B+x)` → different root, same suffix

Primary metric:

```
Δ = sim(A+x, A+y) − sim(A+x, B+x)
```

Interpretation:

- **Δ > 0** → model treats shared root as more important than shared suffix → **root-dominant representation**
- **Δ < 0** → shared suffix pulls embeddings closer than shared root → **suffix-dominant representation**
- **Δ ≈ 0** → balanced

### 1.4 Methodological controls

The experiment must control several Turkish-specific confounds:

1. **Vowel harmony** (vokal uyumu): Turkish suffixes have multiple variants (`-ler` vs `-lar`). Comparing `evler` and `okullar` confounds "different root" with "different suffix form". Solution: build two **parallel sets** — one with front-vowel roots, one with back-vowel roots — and compare within set only.

2. **Consonant assimilation** (konsonant uyumu): After hard consonants (`p ç t k f h s ş`), `-de` becomes `-te`, `-den` becomes `-tan`. Solution: implemented in stimulus generation.

3. **Four-way vowel harmony** for `-lI`, `-lIk`, `-sIz`: Round vowels produce round suffix vowels (`top → toplu`, not `toplı`). Solution: implemented in stimulus generation.

4. **Frequency matching**: A and B roots are matched on word frequency to avoid the "common vs rare" confound. All roots from Turkish Zipf band 4.0–5.6 (medium frequency in `wordfreq` library).

5. **Tokenization fragmentation**: For the probe to make sense, both A+x and A+y must be split into ≥2 sub-tokens by BERTurk. This is verified by the tokenization check step.

---

## 2. What is already done

### 2.1 Stimulus design (complete)

**File: `triplet_stimulus.csv`** — 60 triplets in CSV format.

Structure:

- 16 frequency-matched concrete noun roots
  - 8 front-vowel (`ince`): ev, göz, deniz, şehir, köpek, çiçek, gemi, ekmek
  - 8 back-vowel (`kalın`): kitap, okul, kalp, taş, kapı, ayak, kafa, masa
- 6 suffix types in 3 categories
  - **Inflectional**: `-lAr` (PLURAL), `-DA` (LOCATIVE), `-DAn` (ABLATIVE)
  - **Light derivational**: `-lI` (WITH), `-lIk` (NOMINAL)
  - **Strong derivational**: `-sIz` (WITHOUT)
- Within each vowel class: 5 fixed A–B root pairs
- Total: 2 vowel classes × 6 suffixes × 5 pairs = **60 triplets**

CSV columns:

```
triplet_id, vowel_class, category, primary_suffix, secondary_suffix,
root_A, root_B, A_x, A_y, B_x,
zipf_A, zipf_B, zipf_diff
```

### 2.2 Stimulus generator (complete)

**File: `build_stimulus.py`** — regenerates the stimulus from scratch if needed.

Implements:

- 4-way vowel harmony rule
- 2-way vowel harmony rule
- Consonant assimilation rule (d→t after hard consonants)
- Pair construction (frequency-adjacent)
- Triplet construction (cross-suffix design)

### 2.3 Frequency source (decided)

**Library:** `wordfreq` (Python package by Robert Speer)

**Justification for thesis:** wordfreq combines multiple sources (Wikipedia, OpenSubtitles, Twitter, Common Crawl, Leeds Internet Corpus) using a weighted median, providing a more robust frequency estimate than any single corpus. The Zipf scale used is the standard from SUBTLEX (Brysbaert et al.). All roots fall in medium-frequency band Zipf 4.0–5.6.

**Citation for thesis:**
> Speer, R. (2022). rspeer/wordfreq: v3.0 (Version v3.0.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.7199437
>
> Van Heuven, W. J. B., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK: A new and improved word frequency database for British English. *Quarterly Journal of Experimental Psychology*, 67(6), 1176–1190.

### 2.4 Tokenization check script (complete, not yet run)

**File: `check_tokenization.py`** — must be run on a machine with internet access to download BERTurk, mBERT, and XLM-R tokenizers.

Verifies for each word in each triplet:

- Number of sub-tokens per word per model
- Whether all three words in the triplet are split (≥2 sub-tokens) by BERTurk
- Whether A+x and B+x share the same final sub-token (sanity check on suffix tokenization)

Output: `triplet_stimulus_with_tokenization.csv`

---

## 3. What needs to be implemented

### 3.1 Step 1 — Run tokenization check (HUMAN, ~5 minutes)

**Command:**

```bash
cd thesis-project/  # or wherever the probe lives
python3 check_tokenization.py triplet_stimulus.csv
```

**Expected output:**

- `triplet_stimulus_with_tokenization.csv` with new columns:
  - `A_x_BERTurk_ntokens`, `A_x_BERTurk_tokens`
  - `A_y_BERTurk_ntokens`, `A_y_BERTurk_tokens`
  - `B_x_BERTurk_ntokens`, `B_x_BERTurk_tokens`
  - Same triple for mBERT and XLM-R
  - `BERTurk_all_split` (bool): True if all 3 words are split
  - `BERTurk_shared_final_subtoken` (bool): True if A+x and B+x share the last sub-token

**Decision rule after this step:**

- If ≥80% of triplets are "all split" in BERTurk → proceed with all 60.
- If 60–80% → keep only the "all split" triplets, note exclusions.
- If <60% → revisit suffix selection (suffixes that don't trigger splitting are not informative).

### 3.2 Step 2 — Implement embedding extraction notebook (CODEX TARGET)

**File to create:** `06_triplet_probe_experiment.ipynb`

**Inputs:**

- `triplet_stimulus_with_tokenization.csv` (output of Step 1)
- The same three Hugging Face models used elsewhere:
  - BERTurk: `dbmdz/bert-base-turkish-cased`
  - mBERT: `bert-base-multilingual-cased`
  - XLM-R: `xlm-roberta-base`

**Functionality:**

This notebook should follow the same architecture as `03_all_model_isolated_embedding_experiment.ipynb`, with these adaptations:

#### 3.2.1 Embedding extraction

For each word in `{A_x, A_y, B_x}` across all 60 triplets:

```python
def get_word_embedding(word, model, tokenizer, layer, pooling):
    """
    Encode word in isolation.
    Extract sub-token embeddings from the specified hidden layer.
    Exclude special tokens (CLS, SEP).
    Apply pooling strategy.
    Return single vector of model hidden size.
    """
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Layer indexing: layer 0 = embedding layer, layers 1-12 = transformer blocks
    hidden = outputs.hidden_states[layer]  # shape: [1, n_tokens, hidden_size]
    
    # Remove CLS and SEP (first and last positions for BERT-like)
    # For XLM-R: same structure, <s> and </s>
    word_tokens = hidden[0, 1:-1, :]  # shape: [n_word_tokens, hidden_size]
    
    if pooling == 'first':
        return word_tokens[0]
    elif pooling == 'last':
        return word_tokens[-1]
    elif pooling == 'mean':
        return word_tokens.mean(dim=0)
    elif pooling == 'max':
        return word_tokens.max(dim=0).values
```

**IMPORTANT:** Use the **isolated condition** for this probe (each word fed alone to the model). The probe's logic — root vs suffix attribution — works most cleanly in isolation. Contextual averaging would dilute the signal by mixing in unrelated sentence content. This is also what's testable in the available time.

#### 3.2.2 Similarity computation

For each triplet × model × layer × pooling configuration:

```python
emb_Ax = get_word_embedding(triplet['A_x'], ...)
emb_Ay = get_word_embedding(triplet['A_y'], ...)
emb_Bx = get_word_embedding(triplet['B_x'], ...)

sim_same_root  = cosine_similarity(emb_Ax, emb_Ay)
sim_same_suffix = cosine_similarity(emb_Ax, emb_Bx)
delta = sim_same_root - sim_same_suffix
```

Save to a long-format DataFrame:

| triplet_id | vowel_class | category | primary_suffix | model | layer | pooling | sim_same_root | sim_same_suffix | delta |
|---|---|---|---|---|---|---|---|---|---|
| T001 | ince | inflectional | PLURAL | BERTurk | 1 | first | 0.84 | 0.61 | 0.23 |
| ... | | | | | | | | | |

Total rows: 60 triplets × 3 models × 3 layers × 4 poolings = **2160 rows**.

#### 3.2.3 Summary aggregation

Aggregate over triplets to produce one row per (model × layer × pooling × category × vowel_class):

| model | layer | pooling | category | vowel_class | n | mean_delta | ci95_low | ci95_high | wilcoxon_p |
|---|---|---|---|---|---|---|---|---|---|
| BERTurk | 7 | first | inflectional | ince | 5 | 0.21 | 0.12 | 0.30 | 0.043 |
| ... | | | | | | | | | |

Statistics:

- **Mean Δ** over triplets in cell
- **Bootstrap 95% CI** (1000 resamples, percentile method)
- **Wilcoxon signed-rank test** comparing sim_same_root vs sim_same_suffix paired across triplets

#### 3.2.4 Output files

```
outputs/results/0601-triplet_probe_per_triplet.csv     (long format, 2160 rows)
outputs/results/0602-triplet_probe_summary.csv         (aggregated by cell)
outputs/results/0603-triplet_probe_main_table.csv      (best pooling/layer per model, for paper)
```

#### 3.2.5 Figures

```
outputs/figures/0601-delta_by_model_pooling.png
    → Bar chart: x = pooling strategy, y = mean Δ, grouped by model. Layer 7 only.
    
outputs/figures/0602-delta_heatmap_by_layer_pooling.png  
    → Heatmap: rows = layers (1, 7, 12), cols = pooling (first/last/mean/max).
       Cell color = mean Δ. One panel per model. 
       
outputs/figures/0603-delta_by_category.png
    → Bar chart broken down by suffix category (inflectional / light_deriv / strong_deriv).
       Hypothesis: derivational suffixes pull Δ down (suffix carries semantic weight).
    
outputs/figures/0604-vowel_class_robustness.png
    → Side-by-side ince vs kalın results to verify the effect replicates in both.

outputs/figures/0605-individual_triplet_deltas.png  
    → Scatter plot: each dot is one triplet's Δ. Colored by category.
       Reveals individual variability and outliers.
```

### 3.3 Step 3 — Interpretation logic (for the writeup)

After the experiment, results should fall into one of these patterns:

**Pattern A — Root-dominant (Δ > 0 across the board):**
Model representations are pulled together more by shared stem than by shared suffix. Confirms that compositional sub-token representations preserve stem identity. Supports the use of first-pooling: if stem info lives in the first sub-token, first-pooling captures the dominant signal.

**Pattern B — Suffix-dominant (Δ < 0):**
Unexpected. Would suggest the suffix sub-token is doing more representational work than the stem. Could indicate that mid-frequency Turkish stems are themselves split in ways that fragment their meaning across multiple stem sub-tokens, while suffixes (being short, frequent, predictable) get a clean single sub-token.

**Pattern C — Pooling-dependent (Δ sign varies):**
The "true" answer depends on which pooling is used. First pooling → root-dominant. Last pooling → suffix-dominant. Mean → in between. This would be the most informative outcome and the strongest evidence for the main experiment's choice of pooling.

**Pattern D — Category-dependent:**
Inflectional suffixes preserve root meaning (Δ > 0). Strong derivational suffixes (`-sIz` "without") flip meaning, so even with shared root, similarity drops — Δ near 0 or negative. This would show the model captures the **semantic** difference between functional and derivational morphology.

---

## 4. Implementation guidelines for codex

### 4.1 Code style

- Follow the existing project style — match `03_all_model_isolated_embedding_experiment.ipynb` patterns
- Use `torch.no_grad()` for inference
- Move model to MPS / CUDA / CPU in that priority order (already done in existing notebooks)
- Set random seeds for reproducibility (Python, NumPy, PyTorch)
- Cache models once, iterate over triplets without reloading

### 4.2 Reuse existing utilities

The existing notebooks have utilities for:

- Model + tokenizer loading
- Cosine similarity
- Spearman correlation
- Saving CSVs in UTF-8 with consistent formatting

Reuse these. Do not write new ones if existing ones work.

### 4.3 Handle edge cases

- If a word has only 1 sub-token in some model (rare for the chosen stimulus, but possible): include it in extraction but flag in output. The pooling strategy will collapse to a single vector, so `first == last == mean == max`. Worth recording for transparency.
- If `output_hidden_states=True` returns hidden states with shape `[1, T, D]`, slice correctly: layer 0 is embedding output, layers 1..12 are transformer blocks.

### 4.4 Performance

60 triplets × 3 words = 180 word forms. Across 3 models × 3 layers × 4 poolings, that's 6480 embedding extractions. On CPU this takes <5 minutes. No GPU needed.

Optimization: extract all hidden states for all layers in a single forward pass per (model, word), then apply all four poolings post-hoc. This is 60×3×3 = 540 forward passes instead of 6480.

### 4.5 Reproducibility

- Save the stimulus CSV alongside results
- Save model versions / commit hashes used
- Random seed: use the same seed as the main experiment (likely 42)

---

## 5. Pitfalls to avoid

1. **Do not lowercase the input** before tokenization unless the model is uncased. BERTurk is cased; mBERT (cased) is cased; XLM-R is generally case-sensitive too. The input words in `triplet_stimulus.csv` are already lowercase, which is the natural form for the experiment.

2. **Do not include CLS/SEP in pooling.** Common bug. Slice them out: `hidden[0, 1:-1, :]` for BERT-style, similar for XLM-R (verify by inspecting tokens).

3. **Do not mix vowel classes in a single comparison.** All comparisons must be within the same vowel class. The CSV already enforces this — preserve it.

4. **Do not interpret Δ values without their confidence intervals.** With only 5 triplets per cell, a single outlier can flip the sign. Always report bootstrap CIs.

5. **Do not call this probe the "main result".** Frame it as a *follow-up* / *mechanistic probe* / *exploratory analysis* in the thesis. The main result is the AnlamVer Spearman analysis.

---

## 6. Suggested implementation order

1. ✅ Stimulus generation (DONE)
2. ⬜ Run tokenization check on local machine, share results
3. ⬜ Decide which triplets to keep (probably all if BERTurk splits most of them)
4. ⬜ Implement `06_triplet_probe_experiment.ipynb`:
   - 4.1 Cell 1: Imports, config, seeds
   - 4.2 Cell 2: Load stimulus CSV
   - 4.3 Cell 3: Load all three models + tokenizers
   - 4.4 Cell 4: Embedding extraction function with `output_hidden_states`
   - 4.5 Cell 5: Loop over (triplet × model × layer × pooling), collect per-triplet results
   - 4.6 Cell 6: Save `0601-triplet_probe_per_triplet.csv`
   - 4.7 Cell 7: Aggregation with bootstrap CI + Wilcoxon
   - 4.8 Cell 8: Save `0602-triplet_probe_summary.csv` and `0603-triplet_probe_main_table.csv`
   - 4.9 Cells 9–13: Five figures (see §3.2.5)
5. ⬜ Review results, decide if anything needs filtering or extra analysis
6. ⬜ Write thesis subsection (see §7)

---

## 7. How to write this in the thesis

### 7.1 Where it goes

This probe is a **subsection of Methods + Results + Discussion**, not a separate chapter.

Suggested placement:

```
Chapter 4 — Results
  4.1 Main experiment: Subtoken pooling on AnlamVer (existing)
    4.1.1 Isolated condition
    4.1.2 Contextual condition
    4.1.3 Effect of tokenization fragmentation
  4.2 Follow-up probe: Where does meaning live? (NEW)
    4.2.1 Probe design
    4.2.2 Stimulus construction
    4.2.3 Results: root vs suffix contributions
    4.2.4 Interaction with pooling strategy

Chapter 5 — Discussion
  5.1 Why does first-pooling outperform mean-pooling?
       → cite probe results here
  5.2 Implications for morphologically rich languages
       → probe gives mechanistic support
```

### 7.2 How to introduce the probe

Suggested opening paragraph for §4.2:

> The main experiment showed that first-subtoken pooling outperformed mean and max pooling for BERTurk, the Turkish-specific model. This is counterintuitive: averaging all subtoken vectors should, in principle, preserve more information than picking a single one. To better understand this outcome, we conducted a follow-up probe testing whether the model's representation of a Turkish word relies more heavily on its stem (kök) or its suffix (ek). If the first sub-token tends to carry stem-level information — as is plausible for an agglutinative language where the stem precedes the affixes — first-pooling would amount to a "stem-keeping" strategy and its success in the main experiment becomes interpretable.

### 7.3 Framing in Discussion

> The probe results [insert finding] are consistent with the interpretation that BERTurk's first sub-token preferentially encodes stem-level semantic content. This explains the otherwise surprising advantage of first-pooling on AnlamVer: when the goal is to capture lexical-semantic similarity between bare nouns, the first sub-token already contains most of the relevant information, while subsequent sub-tokens add suffix-level noise.

### 7.4 What to acknowledge as limitations

- **Small N**: 60 triplets is exploratory. Results should be interpreted with confidence intervals, not p-values.
- **Stimulus is morphologically generated, not corpus-attested**: Some forms like `okulluk` are well-formed but unattested. This is intentional — it tests compositional behavior — but is worth noting.
- **Cosine similarity is one metric**: Other metrics (Euclidean, dot-product) could give different patterns. Cosine chosen for consistency with the main experiment.
- **Probe is isolated, not contextual**: Adding context would dilute the probe signal; future work could extend to contextual probes.

---

## 8. References (to add to thesis bibliography)

```
Speer, R. (2022). rspeer/wordfreq: v3.0 (Computer software). Zenodo. https://doi.org/10.5281/zenodo.7199437

Van Heuven, W. J. B., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK: A new and improved word frequency database for British English. Quarterly Journal of Experimental Psychology, 67(6), 1176–1190. https://doi.org/10.1080/17470218.2013.850521

Apidianaki, M. (2023). From Word Types to Tokens and Back: A Survey of Approaches to Word Meaning Representation and Interpretation. Computational Linguistics, 49(2), 465–523. https://doi.org/10.1162/coli_a_00474
```

The Apidianaki review is the one Cassani recommended in Meeting 6, and it covers strategies for deriving word-level embeddings from contextual models — directly relevant background for this probe.

---

## 9. Open questions / decisions to revisit

These may come up during implementation:

- **Should we also run this probe on the contextual condition** (averaging across multiple sentences)? Currently planned as isolated-only for time reasons. If the isolated probe is finished by mid-week, add contextual as a robustness check.
- **Layer selection**: Currently layers 1, 7, 12 (same as main experiment). Could add layer 9 if there's reason to think middle-late layers behave differently for compositionality.
- **Should derivational suffixes be analyzed separately**? In the writeup, yes — break down by category. In the main aggregation, also yes (already planned in §3.2.3).

---

## 10. Quick reference: files

| File | Purpose | Status |
|------|---------|--------|
| `build_stimulus.py` | Regenerates the stimulus CSV | ✅ Done |
| `triplet_stimulus.csv` | 60 triplets, ready to use | ✅ Done |
| `root_inventory.csv` | List of 16 roots + Zipf frequencies | ✅ Done |
| `suffix_inventory.csv` | List of 6 suffixes + examples | ✅ Done |
| `check_tokenization.py` | BERTurk/mBERT/XLM-R tokenization check | ✅ Done (not yet run) |
| `triplet_stimulus_with_tokenization.csv` | Stimulus + sub-token counts per model | ⬜ Output of `check_tokenization.py` |
| `06_triplet_probe_experiment.ipynb` | Main experiment notebook | ⬜ **TO IMPLEMENT** |
| `outputs/results/0601-triplet_probe_per_triplet.csv` | Long-format per-triplet results | ⬜ Output of notebook |
| `outputs/results/0602-triplet_probe_summary.csv` | Aggregated by cell | ⬜ Output of notebook |
| `outputs/results/0603-triplet_probe_main_table.csv` | Headline table | ⬜ Output of notebook |
| `outputs/figures/0601...0605` | Five figures | ⬜ Output of notebook |

---

## 11. Prompt suggestion for codex

When asking codex to implement `06_triplet_probe_experiment.ipynb`, suggested prompt:

> Implement the notebook `06_triplet_probe_experiment.ipynb` according to the plan in `TRIPLET_PROBE_PLAN.md` section 3.2.
>
> Mirror the architecture and style of `03_all_model_isolated_embedding_experiment.ipynb` exactly:
> - Same model loading pattern
> - Same device selection (MPS > CUDA > CPU)
> - Same way of slicing CLS/SEP from hidden states
> - Same CSV output conventions (UTF-8, no index column)
>
> Differences from notebook 03:
> - Input is `triplet_stimulus_with_tokenization.csv` (not AnlamVer)
> - For each triplet, extract 3 word embeddings (A_x, A_y, B_x) instead of pair embeddings
> - Primary output metric is Δ = sim(A_x, A_y) − sim(A_x, B_x)
> - Statistical aggregation includes bootstrap 95% CI and Wilcoxon signed-rank test
> - Generate the 5 figures listed in section 3.2.5
>
> Use the existing models (BERTurk, mBERT, XLM-R), layers (1, 7, 12), and pooling strategies (first, last, mean, max).
>
> Performance optimization: extract all hidden states in a single forward pass per (model, word), then apply all four pooling strategies post-hoc.
>
> Output file numbering follows the project convention: `06xx-` prefix for this notebook's outputs.
