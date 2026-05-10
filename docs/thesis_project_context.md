# Thesis Project Context — Subword Tokenization and Turkish Semantic Similarity

**Student:** Nebi Eren Aygün  
**Programme:** BSc Cognitive Science & Artificial Intelligence, Tilburg University  
**Supervisor:** Giovanni Cassani  
**Working title:** *Where Does Meaning Live in Subword Tokenization? Evaluating Subtoken-to-Word Embedding Composition for Turkish on AnlamVer*  
**Current status:** Proposal resit passed, poster presentation passed, GO received to continue with the thesis project.

---

## 1. Project Overview

This thesis investigates how subword tokenization affects the quality of word-level semantic representations for Turkish in BERT-like language models. Turkish is morphologically rich and agglutinative, meaning that a single word can contain a stem and multiple meaning-bearing or grammar-bearing affixes. Because modern transformer models process text using subword tokenization, Turkish words are often split into multiple sub-tokens. This creates a methodological problem: if a word is represented internally as several sub-token embeddings, how should these sub-token embeddings be combined into one word-level embedding?

The project evaluates different **subtoken-to-word embedding composition strategies** against **human semantic similarity judgments** from the Turkish AnlamVer dataset.

The central idea is:

> If a model’s word-level embeddings are semantically meaningful, then cosine similarity between two model-derived word embeddings should correlate with human similarity ratings for the same word pair.

The thesis therefore compares different ways of producing a word embedding from sub-token embeddings and evaluates which method best matches human judgments.

---

## 2. Core Research Problem

Modern LLMs and BERT-like models do not always represent words as single units. Instead, words may be fragmented into multiple subword tokens. This is especially relevant for Turkish, where words can be morphologically complex.

Example:

```text
evlerimizden → ev ##ler ##imiz ##den
```

This raises several questions:

- Does the semantic meaning of the word mainly live in the first token, often corresponding to the stem?
- Is important semantic information distributed across all sub-tokens?
- Are suffix or later sub-tokens also semantically useful?
- Does the best composition strategy depend on the model layer?
- Does the best strategy depend on whether the word is encoded in isolation or inside sentence contexts?
- Does tokenization fragmentation make word-level semantic representation worse?

---

## 3. Main Research Question

**Which subtoken-to-word embedding composition method best corresponds to human semantic similarity judgments in Turkish?**

A more complete working version:

> Which subtoken-to-word embedding composition strategy best matches human semantic similarity judgments in Turkish, and how does this depend on tokenization fragmentation, model layer, model type, and input representation condition?

---

## 4. Sub-questions

### SQ1 — Pooling strategies

Do different subtoken-to-word composition strategies differ in how well they correspond to AnlamVer human semantic similarity scores?

Strategies:

- First sub-token pooling
- Last sub-token pooling
- Mean pooling
- Max pooling
- Optional exploratory strategy: middle-token pooling for words split into 3 or more sub-tokens

### SQ2 — Fragmentation effect

Does the effect of composition strategy change as the number of sub-tokens per word increases?

In other words:

> Are naive strategies such as first-token or last-token pooling more strongly impaired when words are split into more sub-tokens?

### SQ3 — Input representation condition

Does the quality of word-level semantic representation differ depending on whether the word is encoded:

1. in isolation, or  
2. in multiple sentence contexts and then averaged?

### SQ4 — Layer effect

Which model layers produce the strongest correspondence with human similarity judgments?

Planned layers:

- Layer 1 — relatively static / low-level representation
- Layer 7 — middle layer, expected to capture more transferable semantic information
- Layer 12 — final contextual layer

### SQ5 — Model comparison

Do tokenization and embedding composition patterns differ across models?

Planned models:

- BERTurk
- multilingual BERT / mBERT
- XLM-RoBERTa

If time becomes limited, the model comparison can be reduced to two models. The supervisor advised that models can be dropped if needed, but input conditions and composition strategies should not be dropped.

---

## 5. Hypotheses

### H1 — Mean pooling advantage

Mean pooling is expected to produce higher Spearman correlations with human similarity judgments than first-token or last-token pooling. This is because semantic information may be distributed across sub-tokens rather than concentrated in a single token.

### H2 — Fragmentation hurts naive pooling

As the number of sub-tokens increases, first-token and last-token pooling are expected to become less reliable compared to mean pooling. The reason is that a single sub-token may capture only part of the word’s meaning.

### H3 — Contextual sentence averaging improves deeper-layer representations

For deeper layers, embeddings averaged across sentence contexts are expected to be more stable and semantically meaningful than embeddings from isolated words. BERT-like models are contextual models, so isolated words may not provide enough context for higher layers.

### H4 — Middle layers may perform best

Middle layers, especially around layer 7, are expected to produce better similarity correlations than the final layer because final layers may be more specialized for pretraining objectives and more sensitive to context.

---

## 6. Dataset: AnlamVer

The main dataset is **AnlamVer**, a Turkish word similarity and relatedness dataset.

### Dataset properties

- 500 Turkish word pairs
- Human ratings for semantic similarity
- Human ratings for semantic relatedness
- Around 12 annotators
- Turkish-specific gold standard for intrinsic evaluation of word embeddings

### Main variables used

Expected core columns:

- `W1`: first word
- `W2`: second word
- `Sim`: human semantic similarity score
- `Rel`: human semantic relatedness score

The main evaluation will use **similarity** rather than relatedness, unless relatedness is added as a secondary analysis.

### Why AnlamVer is suitable

AnlamVer provides human judgments for Turkish word pairs, making it suitable for intrinsic evaluation of model-derived word embeddings. Since the project asks whether model representations match human semantic similarity judgments, AnlamVer acts as the gold standard.

---

## 7. Current Tokenization Exploration Findings

An initial notebook has already been used to tokenize AnlamVer words with `dbmdz/bert-base-turkish-cased`.

### Current BERTurk tokenization statistics

| Measurement | Result |
|---|---:|
| Total word pairs | 500 |
| Total word occurrences | 1000 |
| Unique words | 317 |
| Pairs where at least one word is split | 275 / 500 |
| Pairs where both words are single-token | 225 / 500 |
| Pairs where at least one word has 3+ sub-tokens | 109 / 500 |
| Pairs where at least one word has 4 sub-tokens | 29 / 500 |
| Unique split words | 132 |
| Unique words with 3+ sub-tokens | 49 |
| Unique words with 4 sub-tokens | 12 |

### Word occurrence distribution

| Subtoken count | Word occurrences | Percentage |
|---:|---:|---:|
| 1 | 620 | 62.0% |
| 2 | 244 | 24.4% |
| 3 | 106 | 10.6% |
| 4 | 30 | 3.0% |

### Unique word distribution

| Subtoken count | Unique words | Percentage |
|---:|---:|---:|
| 1 | 185 | 58.4% |
| 2 | 83 | 26.2% |
| 3 | 37 | 11.7% |
| 4 | 12 | 3.8% |

### Interpretation

The dataset appears sufficient for the main experiment because more than half of the word pairs contain at least one split word. This supports the feasibility of studying subtoken-to-word composition strategies.

The 3+ token subset is smaller, but still large enough for exploratory analyses such as middle-token pooling or high-fragmentation comparisons.

Important caution:

A preliminary negative correlation between total subtoken count and human similarity should not be interpreted causally. More fragmented word pairs may simply differ in frequency, morphology, concreteness, or semantic relation type. Fragmentation should therefore be treated as a descriptive and moderating variable, not as a direct causal explanation.

---

## 8. Models

### Primary model

The first model to implement should be:

```text
dbmdz/bert-base-turkish-cased
```

This should be the starting point because it is Turkish-specific and directly relevant to the research question.

### Additional models

After the BERTurk pipeline works:

```text
bert-base-multilingual-cased
xlm-roberta-base
```

### Scope decision

The full planned model setup is:

- BERTurk
- mBERT
- XLM-RoBERTa

However, if time becomes limited, the model comparison can be reduced. The most important parts to preserve are:

1. input conditions, and  
2. composition strategies.

Layers and models are easier to add once the embedding extraction pipeline is working.

---

## 9. Input Representation Conditions

A key methodological decision is how to obtain a representation for an isolated word from a contextual language model.

### Condition A — Isolated word

The target word is passed directly to the model.

Example:

```text
kelime
```

Advantages:

- Easy to implement
- Directly matches the dataset format
- No external corpus needed

Limitations:

- BERT-like models are contextual; isolated input may produce weak or unstable higher-layer embeddings
- Especially problematic for deeper layers

### Condition B — Contextual sentence average

The target word is embedded in multiple sentence contexts. The target word representation is extracted from each sentence and then averaged.

Example:

```text
Bu kelime günlük dilde sıkça kullanılır.
...
15+ sentences containing the target word
```

For each target word:

1. collect around 15 sentences containing the word,
2. extract the embedding of the target word in each sentence,
3. compose subtoken embeddings into one word embedding per sentence,
4. average across the sentence-level word embeddings.

Advantages:

- Better matches how contextual models work
- Expected to give more stable word-level semantic representations
- Supervisor specifically highlighted this as important

Limitations:

- Requires finding or generating sentences for each target word
- Target word alignment in sentences must be handled carefully
- More time-consuming

### Decision

Both conditions should be implemented if possible.

The supervisor emphasized that isolated words are easy, but contextual sentences take time and should be planned early. Conditions should not be dropped unless absolutely necessary.

---

## 10. Contextual Sentence Sources

Planned sources for real Turkish sentences:

- OSCAR Turkish subset
- Turkish Wikipedia dump
- CC-100 Turkish

Alternative if corpus extraction becomes too time-consuming:

- Use ChatGPT or Gemini to generate sentences containing each target word

Important rule:

> Do not use the same model being evaluated to generate the contextual sentences.

This avoids circular evaluation. For example, BERTurk should not be used to generate contexts for evaluating BERTurk embeddings. Since BERTurk is not a generative model anyway, this is mostly relevant if using open LLMs.

### Practical recommendation

For the thesis timeline, a practical compromise may be:

1. First complete isolated-word pipeline.
2. Try corpus-based sentence extraction for all target words.
3. If coverage is poor or time is limited, use generated sentences as a controlled fallback.
4. Clearly document the source of sentence contexts.

---

## 11. Embedding Extraction Pipeline

For each model, layer, input condition, and pooling strategy:

### Step 1 — Tokenize words

For each word in AnlamVer:

```text
word → sub-tokens
```

Example:

```text
evlerimizden → ev ##ler ##imiz ##den
```

### Step 2 — Feed input to model

Either:

```text
[CLS] word [SEP]
```

or sentence context:

```text
[CLS] sentence containing target word [SEP]
```

### Step 3 — Extract hidden states

Extract hidden states from selected layers:

- Layer 1
- Layer 7
- Layer 12

Potential implementation detail:

Use Hugging Face Transformers with:

```python
output_hidden_states=True
```

### Step 4 — Identify target sub-token positions

For isolated words this is easy because the input contains only the target word plus special tokens.

For sentence contexts this is harder. The pipeline must identify which sub-token positions correspond to the target word in the sentence.

### Step 5 — Apply subtoken-to-word composition

For the target word’s sub-token embeddings:

- First pooling: use the first sub-token vector
- Last pooling: use the last sub-token vector
- Mean pooling: average all sub-token vectors
- Max pooling: take element-wise max across sub-token vectors
- Optional middle pooling: use the middle sub-token for words with 3+ sub-tokens

### Step 6 — Obtain word embeddings

Each target word receives one vector per:

```text
model × layer × input condition × pooling strategy
```

### Step 7 — Compute word-pair similarity

For each AnlamVer pair:

```text
cosine_similarity(embedding(W1), embedding(W2))
```

### Step 8 — Evaluate against human judgments

Compare model cosine similarities against human similarity scores using Spearman rank correlation:

```text
Spearman ρ(model cosine similarities, human similarity scores)
```

---

## 12. Composition Strategies

### First sub-token pooling

Uses only the first sub-token embedding.

Potential interpretation:

- In Turkish, the first sub-token may often contain the stem or semantic core.
- This may perform well when the first token captures most lexical meaning.

Limitation:

- Ignores suffixes and later sub-token information.

### Last sub-token pooling

Uses only the last sub-token embedding.

Potential interpretation:

- May capture suffix or inflectional information.

Limitation:

- May ignore the stem, which is usually semantically central.

### Mean pooling

Averages all sub-token embeddings.

Potential interpretation:

- Assumes semantic information is distributed across sub-tokens.
- Expected to be robust for fragmented words.

### Max pooling

Takes the maximum value across sub-token embeddings for each vector dimension.

Potential interpretation:

- Captures the strongest activated features across sub-tokens.

Limitation:

- Less directly interpretable than mean pooling.

### Middle-token pooling

Only for 3+ sub-token words.

Potential interpretation:

- Suggested by the supervisor as an additional exploratory strategy.
- Useful to test whether limiting the comparison to first vs last is too narrow.

Decision:

Middle-token pooling should be treated as exploratory because the 3+ sub-token subset is relatively small.

---

## 13. Evaluation Metrics

### Primary metric

```text
Spearman rank correlation coefficient (ρ)
```

Reason:

- Human similarity judgments are ordinal or rating-based.
- The question is whether model similarities rank word pairs similarly to humans.

### Pair-level similarity metric

```text
Cosine similarity
```

Reason:

- Standard measure for comparing embedding vectors.

### Additional descriptive analyses

- Subtoken count distribution
- Clean vs split pair comparison
- Low / medium / high fragmentation bins
- Possibly correlation between fragmentation and model error

Potential model error measure:

```text
absolute_error = abs(rank/model similarity discrepancy)
```

or simpler:

```text
compare Spearman ρ across fragmentation bins
```

---

## 14. Baselines

Possible baselines:

### FastText Turkish vectors

A static word embedding baseline.

Reason:

- FastText uses subword information but produces word-level vectors.
- Useful comparison against contextual BERT-like embeddings.

### Random embeddings

A sanity-check baseline.

Reason:

- Ensures that observed correlations are meaningfully above random.

### First-token pooling

Can also function as a simple baseline composition strategy.

Reason:

- A naive approach often used implicitly in token-level models.

---

## 15. Planned Main Experiment

### Experiment 1 — AnlamVer semantic similarity evaluation

**Goal:** Evaluate which subtoken-to-word composition strategy best aligns with human semantic similarity judgments.

### Independent variables

- Model
  - BERTurk
  - mBERT
  - XLM-R
- Layer
  - 1
  - 7
  - 12
- Input condition
  - isolated word
  - contextual sentence average
- Pooling strategy
  - first
  - last
  - mean
  - max
  - optional middle
- Fragmentation level
  - one token
  - two tokens
  - three or more tokens
  - or pair-level max/total subtoken count

### Dependent variable

- Spearman correlation between model-derived cosine similarities and human similarity scores.

### Expected output

A results table like:

| Model | Layer | Input condition | Pooling | Spearman ρ |
|---|---:|---|---|---:|
| BERTurk | 1 | isolated | first | ... |
| BERTurk | 1 | isolated | mean | ... |
| BERTurk | 7 | contextual | mean | ... |
| ... | ... | ... | ... | ... |

Possible visualizations:

- Bar plot of Spearman ρ by pooling strategy
- Line plot across layers
- Heatmap: layer × pooling
- Split vs clean pair comparison
- Scatter/box plot of fragmentation level vs model error

---

## 16. Optional Second Experiment

### Experiment 2 — Curated Turkish morphology comparison

This was suggested by the supervisor as a way to make the project more linguistically substantial.

### Idea

Create triplets where:

- Word A and Word B share the same stem but have different affixes.
- Word B and Word C share the same affix but have different stems.

Example structure:

```text
A = stem1 + affixA
B = stem1 + affixB
C = stem2 + affixB
```

The expectation is usually:

```text
similarity(A, B) > similarity(B, C)
```

because sharing a stem should generally imply stronger semantic similarity than sharing only an affix.

### Research question for Experiment 2

Do BERT-like models represent stem overlap as more semantically important than affix overlap, and does this depend on tokenization alignment?

### Why it is useful

This experiment would complement the AnlamVer experiment:

- Experiment 1: human similarity benchmark
- Experiment 2: controlled linguistic structure benchmark

### Scope decision

This is optional. It should only be attempted after the main AnlamVer experiment is running.

---

## 17. Current Implementation State

A preliminary notebook exists for BERTurk tokenization exploration.

### What the notebook currently does well

- Loads BERTurk tokenizer
- Reads AnlamVer CSV
- Computes subtoken counts for W1 and W2
- Adds token lists
- Produces basic fragmentation statistics
- Separates clean pairs and split pairs
- Checks rough Spearman relation between token count and human similarity

### Problems to fix

#### 1. Only BERTurk tokenization is included

Need to extend to:

- BERTurk
- mBERT
- XLM-R

Each model needs separate token count columns.

Suggested column names:

```text
berturk_w1_tokens
berturk_w1_n_subtokens
berturk_w2_tokens
berturk_w2_n_subtokens
mbert_w1_tokens
...
xlmr_w1_tokens
...
```

#### 2. Similarity scores should be numeric

The dataset may use comma decimal notation.

Need to convert:

```python
df["Sim_num"] = df["Sim"].astype(str).str.replace(",", ".", regex=False).astype(float)
df["Rel_num"] = df["Rel"].astype(str).str.replace(",", ".", regex=False).astype(float)
```

#### 3. Save derived columns after creating them

In the current notebook, some derived columns such as `total_subtokens` may be created after saving the CSV. This should be fixed.

#### 4. Token lists in CSV are stringified

This is acceptable for inspection, but for reproducible experiments it is better to regenerate tokenization from the tokenizer rather than rely on stringified lists.

#### 5. Spearman between token count and similarity should be interpreted cautiously

This analysis is descriptive. It should not be interpreted as showing that tokenization causes lower human similarity.

---

## 18. Immediate Next Technical Sprint

### Sprint 1 — Thesis-ready tokenization analysis

Goal:

> Produce a clean, reproducible tokenization analysis for all planned models.

Tasks:

1. Load AnlamVer dataset.
2. Normalize column names.
3. Convert `Sim` and `Rel` to numeric values.
4. Extract unique words.
5. Load tokenizers:
   - BERTurk
   - mBERT
   - XLM-R
6. Tokenize each word with each tokenizer.
7. Store token lists and subtoken counts.
8. Compute pair-level fragmentation variables:
   - total subtokens
   - max subtokens
   - whether both words are single-token
   - whether at least one word is split
   - whether at least one word has 3+ sub-tokens
9. Produce summary tables:
   - word-level subtoken distribution per model
   - pair-level split distribution per model
   - top most fragmented words per model
10. Save clean output:

```text
anlamver_tokenization_analysis.csv
```

11. Create figures/tables for thesis Methods or Results.

---

## 19. Second Technical Sprint

### Sprint 2 — Isolated embedding extraction with BERTurk

Goal:

> Implement the full isolated-word embedding pipeline for one model before scaling up.

Tasks:

1. Load BERTurk model with hidden states.
2. For each unique word:
   - tokenize
   - feed isolated word to model
   - extract layers 1, 7, 12
   - identify word subtoken positions
   - apply first, last, mean, max pooling
3. Save embeddings or computed pair similarities.
4. For each word pair:
   - compute cosine similarity
5. For each condition:
   - compute Spearman correlation with `Sim_num`
6. Create first results table.

Output:

```text
berturk_isolated_results.csv
```

This sprint gives the first real thesis result.

---

## 20. Third Technical Sprint

### Sprint 3 — Scale to layers, strategies, and models

Goal:

> Generalize the code so it can run across all models, layers, and pooling strategies.

Tasks:

1. Refactor code into functions:
   - `load_model_and_tokenizer()`
   - `tokenize_word()`
   - `extract_word_embedding()`
   - `apply_pooling()`
   - `compute_pair_similarities()`
   - `evaluate_spearman()`
2. Run full isolated-word experiment for:
   - BERTurk
   - mBERT
   - XLM-R, if time permits
3. Save consolidated result table.

---

## 21. Fourth Technical Sprint

### Sprint 4 — Contextual sentence condition

Goal:

> Implement sentence-based word representations.

Tasks:

1. Build target word list from AnlamVer.
2. Collect or generate around 15 sentences per word.
3. Store sentence file:

```text
word, sentence_id, sentence, source
```

4. For each sentence:
   - tokenize sentence
   - find target word subtoken span
   - extract target word embedding
   - apply pooling
5. Average embeddings across sentences per word.
6. Compute pair similarities.
7. Evaluate Spearman correlation.
8. Compare contextual vs isolated results.

Potential problem:

Target word matching in Turkish sentences can be difficult if the word appears with inflection or punctuation. The first version should use exact word matching to keep the pipeline reliable.

---

## 22. Fifth Technical Sprint

### Sprint 5 — Analysis and visualization

Goal:

> Turn raw result tables into thesis-ready results.

Main analyses:

1. Best pooling strategy overall
2. Best layer overall
3. Isolated vs contextual comparison
4. Model comparison
5. Fragmentation effect
6. Clean vs split pair comparison
7. Optional 3+ token / middle-token analysis

Figures:

- Spearman by pooling strategy
- Spearman by layer
- Heatmap of pooling × layer
- Split vs clean comparison
- Fragmentation bin comparison

Tables:

- Tokenization summary table
- Main results table
- Best configuration table
- Optional appendix table with all model/layer/condition/pooling combinations

---

## 23. Thesis Report Structure

The thesis report should follow the university thesis report guidelines.

### Title page

Working title:

> Where Does Meaning Live in Subword Tokenization?

Possible final title under 15 words:

> Subword Composition and Turkish Semantic Similarity

or:

> Evaluating Turkish Word Embeddings under Subword Tokenization

### Abstract

150–250 words. Should include:

- Problem
- Research goal
- Method
- Dataset
- Main findings

Write this last.

### DSECT statement

Must disclose:

- Data source
- Ethics
- Code
- Technology use
- AI-assisted writing/tool use

Draft statement idea:

> During the preparation of this work, the author used ChatGPT for brainstorming, planning, code debugging, and language revision. The author reviewed and edited all AI-assisted output and takes full responsibility for the final content.

This must be adapted honestly based on actual use.

### Section 1 — Introduction

Needs to explain:

- Why word-level semantic representation matters
- Why subword tokenization creates a problem
- Why Turkish is a good test case
- What the thesis investigates
- Research question and hypotheses
- Very brief method overview
- Brief preview of findings, after results exist

Important supervisor feedback:

The introduction must clearly explain isolated vs contextualized representations if they remain part of the research questions.

### Section 2 — Related Work

Should be organized around the thesis logic, not just paper summaries.

Possible subsections:

1. Subword tokenization in morphologically rich languages
2. Tokenization and psycholinguistic validity
3. From sub-token embeddings to word embeddings
4. Contextualized models as sources of word-level semantic representations
5. Intrinsic evaluation with human similarity judgments

Important writing rule from supervisor:

For each paper, explain why it matters for this thesis:

```text
They do X → I do Y.
They found X → therefore I test Y.
They show limitation X → my project addresses X.
```

### Section 3 — Methods

Should include:

- Dataset description
- Tokenization analysis
- Models
- Input conditions
- Layer selection
- Pooling strategies
- Similarity computation
- Evaluation metric
- Software and reproducibility details

### Section 4 — Results

Should include:

1. Tokenization/fragmentation descriptive results
2. Main Spearman results
3. Pooling strategy comparison
4. Layer comparison
5. Input condition comparison
6. Fragmentation analysis
7. Optional model comparison
8. Optional morphology experiment

### Section 5 — Discussion

Should answer:

- Which composition strategy worked best?
- Did meaning appear concentrated or distributed across sub-tokens?
- Did fragmentation matter?
- Did contextual sentence averaging help?
- How do results relate to prior work?
- What are the limitations?

### Section 6 — Conclusion

Short final answer to the research question.

---

## 24. Key Literature Already in Proposal

The proposal already includes these references:

- Ács (2021) — subword pooling strategies
- Cassani et al. (2023) — stability of BERT embeddings for psycholinguistic research
- Conneau et al. (2020) — XLM-R / cross-lingual representations
- Devlin et al. (2019) — BERT
- Ercan & Yıldız (2018) — AnlamVer
- Giulianelli et al. (2024) — tokenization in psycholinguistics
- Kaya & Tantuğ (2024) — tokenization granularity for Turkish LLMs
- Li et al. (2020) — sentence embeddings from pretrained language models
- Ortiz Suárez et al. (2019) — OSCAR corpus
- Schweter (2020) — BERTurk
- Toraman et al. (2023) — impact of tokenization for Turkish
- Xia et al. (2021) — layer/semantic representation motivation

Need to verify and possibly strengthen:

- Haslett & Cai (2025), if used
- Apidianaki, *From Tokens to Words and Back*, recommended by supervisor
- Bomani et al. (2017), recommended by supervisor

---

## 25. Supervisor Feedback to Keep in Mind

### Proposal feedback

- Formatting had to be fixed.
- A milestones table was needed.
- The original idea was clear but not highly ambitious.
- The project needed more substance beyond basic pooling and correlation.
- Important factors to consider:
  - embedding layer
  - isolated vs sentence-context input
  - morphological-token alignment
  - whether enough words are actually split into multiple tokens
- Possible add-ons:
  - embedding marginalization
  - synthetic or curated morpheme-based examples

### Poster / GO feedback

- Poster was well designed.
- Presentation was fairly good.
- Some questions were not answered convincingly.
- The introduction and literature review must explain clearly what the project aims to do and why.
- Final decision: GO, thesis can proceed.

### Later supervisor advice

- The project is not too complex, but it will take time to run everything.
- Prioritize contextual sentence collection because it is the time-consuming part.
- Composition strategies are central and must be implemented and validated.
- Layer extraction is trivial once the embedding pipeline works.
- Model count can be reduced if needed.
- Do not drop input conditions or composition strategies.
- The curated stem-affix dataset would be a second experiment, complementary to AnlamVer.

---

## 26. Scope Priorities

### Minimum viable thesis

Must include:

1. AnlamVer dataset
2. BERTurk model
3. Tokenization fragmentation analysis
4. Isolated word condition
5. Layers 1, 7, 12
6. First / last / mean / max pooling
7. Cosine similarity
8. Spearman evaluation against human similarity
9. Results and discussion

### Stronger thesis

Add:

1. Contextual sentence average condition
2. mBERT comparison
3. XLM-R comparison
4. Fragmentation-bin analysis
5. Mid-token exploratory analysis

### Ambitious thesis

Add:

1. Curated stem-affix dataset
2. Separate morphology experiment
3. More detailed linguistic analysis of token-morpheme alignment

---

## 27. Main Risks

### Risk 1 — Sentence collection takes too long

Mitigation:

- Start with isolated condition.
- Build sentence collection early.
- Use generated sentences as fallback if corpus extraction fails.

### Risk 2 — Too many combinations

Full setup:

```text
3 models × 3 layers × 2 input conditions × 4 pooling strategies
= 72 configurations
```

This is manageable computationally if the pipeline is clean, but can become overwhelming for reporting.

Mitigation:

- First run BERTurk only.
- Use mBERT/XLM-R after the pipeline is stable.
- Put full tables in appendix; report main patterns in the Results section.

### Risk 3 — Target word alignment in sentences is difficult

Mitigation:

- Start with exact word matching.
- Avoid inflected sentence forms in generated sentences.
- Store all sentence-word matches for inspection.

### Risk 4 — Results may be weak or inconsistent

Mitigation:

- Weak results are still meaningful if interpreted carefully.
- The thesis can discuss limitations of BERT-like models for isolated Turkish word semantics.
- Fragmentation and layer analyses can explain why results differ.

### Risk 5 — Literature review becomes a list of summaries

Mitigation:

Use explicit argumentative links:

```text
Ács (2021) shows that pooling matters; therefore, this thesis compares pooling strategies for Turkish.
Cassani et al. (2023) show that isolated BERT embeddings may be unstable; therefore, this thesis compares isolated and contextualized representations.
Giulianelli et al. (2024) argue that tokenization must be handled carefully in psycholinguistic work; therefore, this thesis treats subtoken composition as a central methodological issue.
```

---

## 28. Recommended File Structure

Possible project structure:

```text
thesis_project/
│
├── data/
│   ├── raw/
│   │   └── anlamver.csv
│   ├── processed/
│   │   ├── anlamver_clean.csv
│   │   ├── anlamver_tokenization_analysis.csv
│   │   └── contextual_sentences.csv
│   └── results/
│       ├── tokenization_summary.csv
│       ├── isolated_results.csv
│       ├── contextual_results.csv
│       └── all_results.csv
│
├── notebooks/
│   ├── 01_tokenization_analysis.ipynb
│   ├── 02_berturk_isolated_embeddings.ipynb
│   ├── 03_full_model_comparison.ipynb
│   └── 04_results_visualization.ipynb
│
├── src/
│   ├── data_loading.py
│   ├── tokenization.py
│   ├── embeddings.py
│   ├── pooling.py
│   ├── evaluation.py
│   └── visualization.py
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── thesis/
│   ├── intro_draft.md
│   ├── related_work_notes.md
│   ├── methods_draft.md
│   └── results_notes.md
│
└── README.md
```

---

## 29. Recommended Coding Principles

- Keep raw data unchanged.
- Save processed data separately.
- Use fixed random seeds where randomness exists.
- Log model name, tokenizer name, layer, pooling strategy, and input condition for every result.
- Store all result tables in long format.
- Avoid manually editing result CSVs.
- Validate pooling functions on simple examples before running the full experiment.
- Save package versions for reproducibility.

Suggested long-format result table:

| pair_id | w1 | w2 | human_sim | model | layer | input_condition | pooling | cosine_similarity |
|---|---|---|---:|---|---:|---|---|---:|

Suggested summary table:

| model | layer | input_condition | pooling | n_pairs | spearman_rho | p_value |
|---|---:|---|---|---:|---:|---:|

---

## 30. Next Action Plan

### Immediate next step

Clean and extend the tokenization notebook.

The goal is to produce:

```text
01_tokenization_analysis.ipynb
anlamver_tokenization_analysis.csv
```

### After that

Implement BERTurk isolated embedding extraction.

### Then

Compute first real Spearman results.

### Then

Scale to:

- more layers
- more pooling strategies
- more models
- contextual sentence condition

---

## 31. Current Best Strategic Decision

The project should not jump directly into all models and all conditions at once.

The safest path is:

1. Tokenization analysis across all tokenizers
2. BERTurk isolated condition
3. Full pooling comparison for BERTurk
4. Layer comparison for BERTurk
5. Add mBERT / XLM-R
6. Add contextual sentence condition
7. Add optional morphology experiment only if time remains

This preserves the core thesis while reducing the risk of getting stuck.

---

## 32. One-sentence Thesis Summary

This thesis evaluates how different methods for composing sub-token embeddings into word-level representations affect the alignment between BERT-like model similarities and human semantic similarity judgments for Turkish.

---

## 33. Short Version for Supervisor or README

This project investigates how subword tokenization affects word-level semantic representations in Turkish. Using the AnlamVer dataset of 500 Turkish word pairs with human similarity ratings, the study compares several subtoken-to-word composition strategies — first-token, last-token, mean pooling, max pooling, and optionally middle-token pooling — across BERTurk, mBERT, and XLM-R. Word-pair cosine similarities derived from model embeddings are evaluated against human similarity judgments using Spearman correlation. The project also examines whether performance depends on tokenization fragmentation, embedding layer, and whether words are represented in isolation or averaged across sentence contexts. The main goal is to identify which composition strategy produces the most human-like semantic representations for Turkish and to better understand where word meaning is encoded when morphologically complex words are split into sub-tokens.
