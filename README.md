# Subword Tokenization and Turkish Semantic Similarity

This repository contains the code, data products, and analysis notebooks for a thesis project on subword tokenization and word-level semantic representation in Turkish transformer models.

The project asks how Turkish words that are split into multiple subtokens should be represented as single word embeddings, and whether different subtoken pooling strategies align with human semantic similarity judgments from the AnlamVer dataset.

## Research Goal

Modern BERT-like language models do not always process words as whole units. They often split morphologically complex Turkish words into smaller subtokens. For example, a Turkish word may be represented as a stem-like first subtoken plus one or more suffix-like continuation subtokens. This creates a practical and theoretical problem: when a word is split into several model tokens, which token vector should represent the word?

This project evaluates several subtoken-to-word composition strategies:

- `first`: use the first subtoken embedding.
- `last`: use the last subtoken embedding.
- `mean`: average all subtoken embeddings.
- `max`: take the element-wise maximum across subtoken embeddings.

The central evaluation idea is simple:

1. Convert each Turkish word into a model-derived word embedding.
2. Compute cosine similarity for each AnlamVer word pair.
3. Compare model similarities with human similarity scores using Spearman rank correlation.

If a model representation captures word meaning in a way that resembles human semantic judgments, its cosine similarity rankings should correlate with the AnlamVer `Sim` scores.

## Research Questions

The main research question is:

> Which subtoken-to-word embedding composition strategy best matches human semantic similarity judgments in Turkish?

The project also examines:

- Whether tokenization fragmentation affects semantic similarity performance.
- Whether isolated-word embeddings differ from contextual sentence-averaged embeddings.
- Whether the best pooling strategy changes across transformer layers.
- Whether Turkish-specific and multilingual transformer models behave differently.
- Whether a follow-up morphological triplet probe supports the interpretation that first-subtoken pooling preserves stem/root information.

## Models

The analysis currently compares three Hugging Face transformer models:

| Short name | Hugging Face model | Notes |
|---|---|---|
| `BERTurk` | `dbmdz/bert-base-turkish-cased` | Turkish-specific BERT model |
| `mBERT` | `bert-base-multilingual-cased` | Multilingual WordPiece model |
| `XLM-R` | `xlm-roberta-base` | Multilingual SentencePiece model |

For each model, the main experiments evaluate transformer layers `1`, `7`, and `12`.

## Dataset

The core dataset is AnlamVer, a Turkish semantic similarity and relatedness benchmark.

The raw dataset is stored at:

```text
data/raw/anlamver-final.csv
```

The main columns used by the experiments are:

| Column | Meaning |
|---|---|
| `W1` | First Turkish word in the pair |
| `W2` | Second Turkish word in the pair |
| `Sim` | Human semantic similarity score |
| `Rel` | Human semantic relatedness score |

The primary evaluation uses `Sim`, not `Rel`.

The raw CSV uses the original AnlamVer formatting, including semicolon separators, Windows Turkish encoding, and comma decimals. The notebooks read it with explicit parsing settings where needed and save cleaned UTF-8 CSV outputs under `data/processed/` and `outputs/`.

## Project Structure

```text
.
|-- 01_tokenization_analysis.ipynb
|-- 02_isolated_embedding_experiment.ipynb
|-- 03_all_model_isolated_embedding_experiment.ipynb
|-- 04_generate_llm_context_sentences.ipynb
|-- 05_contextual_embedding_experiment.ipynb
|-- 06_triplet_probe_experiment.ipynb
|-- data/
|   |-- raw/
|   |   `-- anlamver-final.csv
|   `-- processed/
|       |-- anlamver_tokenization_analysis.csv
|       |-- context_sentences_llm.csv
|       |-- context_sentences_llm_invalid.csv
|       |-- context_sentences_llm_raw_batch5_sent15.jsonl
|       `-- triplet_stimulus_with_tokenization.csv
|-- docs/
|   `-- thesis_project_context.md
|-- outputs/
|   |-- figures/
|   |-- results/
|   `-- tables/
|-- triplet_files/
|   |-- build_stimulus.py
|   |-- check_tokenization.py
|   |-- root_inventory.csv
|   |-- suffix_inventory.csv
|   `-- triplet_stimulus.csv
|-- requirements.txt
`-- README.md
```

## Notebook Pipeline

Run the notebooks from the repository root so all relative paths resolve correctly.

### 1. Tokenization Analysis

Notebook:

```text
01_tokenization_analysis.ipynb
```

Purpose:

- Load and clean the raw AnlamVer CSV.
- Extract all unique words from `W1` and `W2`.
- Tokenize words with BERTurk, mBERT, and XLM-R.
- Measure subtoken count, split status, subtoken length, and pair-level tokenization complexity.
- Save processed data and summary tables.

Main outputs:

```text
data/processed/anlamver_tokenization_analysis.csv
outputs/tables/0101-tokenization_subtoken_count_distribution.csv
outputs/tables/0102-tokenization_split_pair_percentage.csv
outputs/tables/0103-tokenization_three_plus_token_words.csv
outputs/tables/0104-tokenization_most_fragmented_words.csv
outputs/tables/0105-tokenization_unique_words.csv
outputs/tables/0106-tokenization_pair_complexity_summary.csv
```

### 2. BERTurk Isolated-Word Pilot

Notebook:

```text
02_isolated_embedding_experiment.ipynb
```

Purpose:

- Validate the embedding extraction workflow on BERTurk only.
- Encode each word in isolation.
- Exclude special tokens such as `[CLS]` and `[SEP]`.
- Pool subtoken embeddings into one word embedding.
- Compute pairwise cosine similarities.
- Evaluate against AnlamVer `Sim` using Spearman correlation.

Main output:

```text
outputs/results/0201-isolated_berturk_results.csv
```

This notebook is mainly a controlled pilot. The all-model isolated experiment below is the main isolated-condition analysis.

### 3. All-Model Isolated Embedding Experiment

Notebook:

```text
03_all_model_isolated_embedding_experiment.ipynb
```

Purpose:

- Run the isolated-word embedding pipeline for BERTurk, mBERT, and XLM-R.
- Compare layers `1`, `7`, and `12`.
- Compare `first`, `last`, `mean`, and `max` pooling.
- Save both pair-level cosine similarities and summary Spearman correlations.
- Join tokenization complexity features onto the pair-level output.
- Generate figures for model, layer, pooling, and fragmentation analyses.

Main outputs:

```text
outputs/results/0301-isolated_all_models_pair_similarities.csv
outputs/results/0302-isolated_all_models_summary.csv
outputs/results/0303-isolated_all_models_pair_similarities_with_tokenization.csv
outputs/figures/0301-model_layer_spearman_heatmap.png
outputs/figures/0302-model_pooling_spearman_heatmap.png
outputs/figures/0303-full_configuration_spearman_heatmap.png
outputs/figures/0304-clean_vs_split_pairs_spearman.png
outputs/figures/0305-fragmentation_level_spearman.png
outputs/figures/0306-spearman_by_pair_total_subtoken_count.png
outputs/figures/0307-spearman_by_pair_max_subtoken_count.png
outputs/figures/0308-clean_minus_split_delta.png
outputs/figures/0309-human_vs_model_similarity_by_complexity.png
outputs/figures/0310-cosine_distribution_clean_vs_split.png
```

### 4. Context Sentence Generation

Notebook:

```text
04_generate_llm_context_sentences.ipynb
```

Purpose:

- Generate controlled Turkish context sentences for every unique target word.
- Require each target word to appear exactly once and without suffixation or spelling changes.
- Save raw LLM responses as JSONL for auditability and resumability.
- Parse and validate generated sentences.
- Export valid and invalid sentence rows.

Main outputs:

```text
data/processed/context_sentences_llm_raw_batch5_sent15.jsonl
data/processed/context_sentences_llm.csv
data/processed/context_sentences_llm_invalid.csv
```

This notebook uses an OpenAI-compatible API endpoint through the `openai` Python SDK. It expects an API key in the environment:

```text
OPENCODE_API_KEY="..."
```

The notebook can also read this value from a local `.env` file in the project root. Do not commit real API keys.

### 5. Contextual Embedding Experiment

Notebook:

```text
05_contextual_embedding_experiment.ipynb
```

Purpose:

- Load generated Turkish context sentences.
- Find the target word span in each sentence with Turkish-aware exact matching.
- Use fast tokenizer offset mappings to identify target subtokens in full sentence context.
- Extract contextual target-word embeddings.
- Average sentence-level target embeddings into one contextual word embedding.
- Compute AnlamVer pair similarities and Spearman correlations.
- Compare contextual results against isolated-word results.

Main outputs:

```text
outputs/results/0501-contextual_all_models_pair_similarities.csv
outputs/results/0502-contextual_all_models_summary.csv
outputs/results/0503-contextual_all_models_pair_similarities_with_tokenization.csv
outputs/results/0504-isolated_vs_contextual_summary.csv
outputs/figures/0501-isolated_vs_contextual_best_by_model.png
outputs/figures/0502-contextual_model_pooling_spearman_heatmap.png
outputs/figures/0503-contextual_full_configuration_spearman_heatmap.png
outputs/figures/0504-contextual_minus_isolated_by_configuration.png
```

Some `040x` contextual result files may also exist in `outputs/results/` and `outputs/figures/` from earlier contextual-analysis runs. The current contextual notebook writes the `050x` files listed above.

### 6. Morphological Triplet Probe

Notebook:

```text
06_triplet_probe_experiment.ipynb
```

Purpose:

- Run a follow-up isolated-word probe testing whether model embeddings are pulled more by shared Turkish roots or shared suffixes.
- Use triplets of the form `A+x`, `A+y`, and `B+x`.
- Compute `sim_same_root = cos(A+x, A+y)` and `sim_same_suffix = cos(A+x, B+x)`.
- Use `delta = sim_same_root - sim_same_suffix` as the primary metric.
- Compare BERTurk, mBERT, and XLM-R over layers `1`, `7`, and `12`.
- Compare `first`, `last`, `mean`, and `max` pooling with special tokens excluded.
- Add robustness checks for stricter tokenization subsets, including BERTurk all-split triplets.

Inputs:

```text
data/processed/triplet_stimulus_with_tokenization.csv
```

If this file is missing, regenerate it from the triplet stimulus:

```bash
python triplet_files/check_tokenization.py triplet_files/triplet_stimulus.csv
cp triplet_files/triplet_stimulus_with_tokenization.csv data/processed/triplet_stimulus_with_tokenization.csv
```

Main outputs:

```text
outputs/results/0601-triplet_probe_per_triplet.csv
outputs/results/0602-triplet_probe_summary.csv
outputs/results/0603-triplet_probe_main_table.csv
outputs/results/0604-triplet_probe_robustness.csv
outputs/figures/0601-delta_by_model_pooling.png
outputs/figures/0602-delta_heatmap_by_layer_pooling.png
outputs/figures/0603-delta_by_category.png
outputs/figures/0604-vowel_class_robustness.png
outputs/figures/0605-individual_triplet_deltas.png
```

This notebook is a mechanistic follow-up to the main AnlamVer experiments. It should be interpreted as exploratory evidence about why first pooling works well, not as a replacement for the main Spearman-correlation analysis.

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name thesis-subword-tokenization
```

## Dependencies

The project uses:

- `pandas` and `numpy` for data processing.
- `scipy` for cosine similarity and Spearman correlation.
- `torch` for model inference.
- `transformers` for Hugging Face tokenizers and models.
- `matplotlib` and `seaborn` for figures.
- `openai` for the OpenAI-compatible context sentence generation endpoint.
- `notebook` and `ipykernel` for Jupyter execution.

See `requirements.txt` for version constraints.

## Running the Full Analysis

Recommended execution order:

```text
01_tokenization_analysis.ipynb
02_isolated_embedding_experiment.ipynb
03_all_model_isolated_embedding_experiment.ipynb
04_generate_llm_context_sentences.ipynb
05_contextual_embedding_experiment.ipynb
06_triplet_probe_experiment.ipynb
```

For a full rerun:

1. Start from the project root.
2. Make sure `data/raw/anlamver-final.csv` exists.
3. Run the tokenization notebook first.
4. Run the isolated embedding notebooks.
5. Set `OPENCODE_API_KEY` before generating context sentences.
6. Run the contextual embedding notebook after context sentences are available.
7. Generate `data/processed/triplet_stimulus_with_tokenization.csv` if needed.
8. Run the triplet probe notebook as the final follow-up analysis.

Model inference can take several minutes depending on hardware and whether Hugging Face model weights are already cached. The notebooks prefer Apple Silicon MPS when available, then CUDA, then CPU.

## Current Output Snapshot

The current repository includes generated intermediate and final outputs.

Tokenization split-pair rates:

| Model | Clean pairs | Split pairs | Split pair percentage |
|---|---:|---:|---:|
| BERTurk | 225 | 275 | 55.0% |
| mBERT | 12 | 488 | 97.6% |
| XLM-R | 93 | 407 | 81.4% |

Best isolated result currently present:

| Input | Model | Layer | Pooling | Spearman rho | Pairs |
|---|---|---:|---|---:|---:|
| isolated | BERTurk | 1 | first | 0.4267 | 500 |

Best contextual result currently present:

| Input | Model | Layer | Pooling | Spearman rho | Pairs |
|---|---|---:|---|---:|---:|
| contextual | BERTurk | 7 | first | 0.5275 | 500 |

Triplet probe headline results currently present:

| Robustness subset | Model | Layer | Pooling | Mean delta | Triplets |
|---|---|---:|---|---:|---:|
| all triplets | BERTurk | 1 | first | 0.3699 | 60 |
| all triplets | XLM-R | 1 | first | 0.3869 | 60 |
| all triplets | mBERT | 1 | first | 0.6366 | 60 |
| BERTurk all-split only | BERTurk | 1 | first | 0.6305 | 13 |

The triplet probe results are root-dominant in the current outputs: words sharing the same root are more similar than words sharing only the same suffix. The BERTurk all-split subset is small, so the probe should be described as exploratory robustness evidence.

These values are a snapshot of the checked-in/generated CSV outputs, not hard-coded conclusions. Re-run the notebooks after changing data, sentence generation, model settings, layers, or pooling strategies.

## Methodological Notes

### Isolated condition

In the isolated condition, the target word is passed to the model alone. Special tokens are removed from the pooling step, and only the actual word subtokens are used. This condition is simple, controlled, and directly tied to the AnlamVer word-pair format.

### Contextual condition

In the contextual condition, each target word is embedded in multiple generated Turkish sentences. The model encodes the full sentence, target subtokens are identified through tokenizer offset mappings, and the sentence-level target embeddings are averaged into one word-level vector.

The contextual condition is designed to better match the way BERT-like models normally operate, because their hidden states are context-sensitive.

### Fragmentation analysis

The project treats tokenization fragmentation as a descriptive and moderating variable. A split word or a high subtoken count does not automatically cause lower semantic performance, but it can help explain when and where pooling strategies become less reliable.

### Triplet probe

The triplet probe is an isolated-word follow-up analysis. It does not use AnlamVer human ratings directly. Instead, it constructs controlled Turkish triplets and asks whether shared root identity or shared suffix identity contributes more to model similarity.

For each triplet:

```text
A+x  and  A+y  share the same root.
A+x  and  B+x  share the same suffix.
delta = cos(A+x, A+y) - cos(A+x, B+x)
```

Positive `delta` values indicate root-dominant representations. Negative values would indicate suffix-dominant representations. The current generated results are positive across the main model summaries and remain positive under stricter tokenization robustness filters.

A key caveat is that BERTurk splits all three words in only a minority of the current 60 triplets. For that reason, the probe is best framed as mechanistic and exploratory: it supports the interpretation that first pooling captures stem/root information, but the main thesis result remains the AnlamVer correlation analysis.

### Evaluation metric

The main metric is Spearman rank correlation between:

- human AnlamVer similarity scores, and
- model-derived cosine similarities.

Spearman correlation is used because the experiment asks whether model similarities preserve the human ranking of word-pair similarity.

## Reproducibility

The main embedding notebooks set random seeds for Python, NumPy, and PyTorch where relevant. Model inference is deterministic enough for the intended analysis, but small numerical differences may occur across devices, PyTorch versions, and transformer backend settings.

The context sentence generation notebook is less deterministic because it calls an external LLM endpoint. To keep this step auditable, it saves raw JSONL responses and parsed validation outputs. Downstream contextual experiments should use the saved `data/processed/context_sentences_llm.csv` file for stable analysis.

## Data and Secret Handling

- Keep raw and processed data files in `data/`.
- Keep generated result tables and figures in `outputs/`.
- Keep research notes and project context in `docs/`.
- Store API keys in `.env` or the shell environment.
- Do not commit real secrets.

## Related Project Notes

Additional thesis context, research motivation, hypotheses, and planned analyses are documented in:

```text
docs/thesis_project_context.md
```
