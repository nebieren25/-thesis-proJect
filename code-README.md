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
|-- 07_results_synthesis_analysis.ipynb
|-- 08_final_results_figures.ipynb
|-- 09_random_embedding_baseline.ipynb
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
`-- code-README.md
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
- Recover case-insensitive exact matches where the generated sentence preserves the target word but changes case.
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

The current contextual run loads `4,755` generated context rows and uses `4,684` case-insensitive exact-match rows. All `317/317` unique AnlamVer target words have at least one usable context row in the current output snapshot.

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

### 7. Results Synthesis Analysis

Notebook:

```text
07_results_synthesis_analysis.ipynb
```

Purpose:

- Read the completed `010x`-`060x` output tables.
- Treat the `050x` contextual outputs as the canonical contextual run.
- Ignore older `040x` contextual outputs except as legacy files from earlier runs.
- Validate expected result-table shapes and model/layer/pooling grids.
- Produce thesis-ready summary tables and figures.
- Summarize the main findings, hypothesis status, and caveats for thesis writing.

Main outputs:

```text
outputs/tables/0701-main_results_for_thesis.csv
outputs/tables/0702-best_by_model_condition.csv
outputs/tables/0703-contextual_gain_summary.csv
outputs/tables/0704-pooling_layer_model_summary.csv
outputs/tables/0705-tokenization_fragmentation_summary.csv
outputs/tables/0706-triplet_probe_thesis_summary.csv
outputs/tables/0707-clean_vs_split_spearman_summary.csv
outputs/figures/0701-main_results_overview.png
outputs/figures/0702-contextual_gain_by_model.png
outputs/figures/0703-triplet_probe_exploratory_summary.png
outputs/figures/0704-tokenization_fragmentation_overview.png
outputs/figures/0705-clean_vs_split_spearman.png
```

The current workspace also contains a detailed configuration-level table:

```text
outputs/tables/0708-detailed_all_results_with_fragmentation.csv
```

This table combines full Spearman results with clean/split, one-word-split, both-words-split, contextual-delta, context-count, and ranking fields. Use it when a thesis table needs one row per full model/layer/pooling/input configuration.

This notebook does not load transformer models or call an API. It is the best starting point for writing the thesis results section because it consolidates the final project outputs into a smaller set of tables and figures.

### 8. Final Results Figures

Notebook:

```text
08_final_results_figures.ipynb
```

Purpose:

- Read completed isolated, contextual, and triplet result CSVs.
- Validate that isolated/contextual pair grids, human scores, and fragmentation labels still match.
- Create mutually exclusive tokenization-fragmentation categories: `clean`, `one_word_split`, and `both_words_split`.
- Rank the best overall configurations and the best configuration for each model/input condition.
- Recompute Spearman correlations within fragmentation categories from pair-level cosine similarities.
- Compute contextual-minus-isolated deltas for every fixed model/layer/pooling configuration.
- Add explicit caveat columns to the triplet-probe summary so it is framed as exploratory evidence.
- Write final thesis-ready `0801-*` tables and figures without overwriting earlier `010x`-`070x` outputs.

Main outputs:

```text
outputs/tables/0801-best_overall_configurations.csv
outputs/tables/0801-best_isolated_vs_contextual_configurations.csv
outputs/tables/0801-clean_one_both_spearman_summary.csv
outputs/tables/0801-fragmentation_category_strategy_summary.csv
outputs/tables/0801-top3_spearman_by_fragmentation_model_condition.csv
outputs/tables/0801-contextual_minus_isolated_delta_by_configuration.csv
outputs/tables/0801-triplet_probe_summary_with_caveats.csv
outputs/figures/0801-best_spearman_by_model_condition.png
outputs/figures/0801-spearman_by_fragmentation_category.png
outputs/figures/0801-contextual_minus_isolated_delta_by_configuration.png
outputs/figures/0801-triplet_same_root_vs_same_suffix_summary.png
```

Additional current clean/split compatibility output:

```text
outputs/tables/0802-clean_split_spearman_summary.csv
```

This notebook does not run transformer inference, embedding extraction, context generation, or API calls. It is the latest compact final-results step for thesis figures and tables.

### 9. Random Embedding Baseline

Notebook:

```text
09_random_embedding_baseline.ipynb
```

Purpose:

- Provide a minimal chance-level lower-bound sanity check for the AnlamVer similarity task.
- Load the cleaned AnlamVer pair data from `data/processed/anlamver_tokenization_analysis.csv` when available.
- Fall back to `data/raw/anlamver-final.csv` using the same raw CSV handling as the main notebooks: `encoding="cp1254"`, `sep=";"`, and comma decimals.
- Use only `W1`, `W2`, and `Sim`.
- Assign each unique word a fixed-seed random 768-dimensional vector.
- Compute cosine similarity for each AnlamVer pair.
- Compare random cosine similarities with human `Sim` scores using Spearman correlation.

Main outputs:

```text
outputs/results/0901-random_baseline_pair_similarities.csv
outputs/results/0902-random_baseline_summary.csv
```

Current checked result:

| Baseline | Embedding dim | Seed | Words | Pairs | Spearman rho | p-value |
|---|---:|---:|---:|---:|---:|---:|
| `random_baseline` | 768 | 42 | 317 | 500 | 0.0093 | 0.8350 |

The result is close to zero and is safe to report as a chance-level lower-bound baseline. It should remain clearly separated from the transformer model results and should not be integrated into the main model comparison tables unless explicitly labelled as `random_baseline`.

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
07_results_synthesis_analysis.ipynb
08_final_results_figures.ipynb
09_random_embedding_baseline.ipynb
```

For a full rerun:

1. Start from the project root.
2. Make sure `data/raw/anlamver-final.csv` exists.
3. Run the tokenization notebook first.
4. Run the isolated embedding notebooks.
5. Set `OPENCODE_API_KEY` before generating context sentences.
6. Run the contextual embedding notebook after context sentences are available.
7. Generate `data/processed/triplet_stimulus_with_tokenization.csv` if needed.
8. Run the triplet probe notebook as the mechanistic follow-up analysis.
9. Run the results synthesis notebook to create thesis-ready `070x` tables and figures.
10. Run the final results figures notebook to create the latest additive `0801` tables and figures.
11. Run the random embedding baseline notebook only as a separate lower-bound sanity check. Do not merge it into the main transformer result tables unless it is clearly labelled as `random_baseline`.

Model inference can take several minutes depending on hardware and whether Hugging Face model weights are already cached. The notebooks prefer Apple Silicon MPS when available, then CUDA, then CPU.

## Current Output Snapshot

The current repository includes generated intermediate, experiment, synthesis, and final-results outputs. For thesis writing and for handing the project to another LLM, start with the `0801` final-results outputs for compact thesis tables/figures, then inspect the `070x` synthesis outputs and earlier notebooks only when a methodological detail is needed.

Latest final-results files:

```text
outputs/tables/0801-best_overall_configurations.csv
outputs/tables/0801-best_isolated_vs_contextual_configurations.csv
outputs/tables/0801-clean_one_both_spearman_summary.csv
outputs/tables/0801-fragmentation_category_strategy_summary.csv
outputs/tables/0801-top3_spearman_by_fragmentation_model_condition.csv
outputs/tables/0801-contextual_minus_isolated_delta_by_configuration.csv
outputs/tables/0801-triplet_probe_summary_with_caveats.csv
outputs/tables/0802-clean_split_spearman_summary.csv
outputs/figures/0801-best_spearman_by_model_condition.png
outputs/figures/0801-spearman_by_fragmentation_category.png
outputs/figures/0801-contextual_minus_isolated_delta_by_configuration.png
outputs/figures/0801-triplet_same_root_vs_same_suffix_summary.png
```

Separate chance-level baseline files:

```text
outputs/results/0901-random_baseline_pair_similarities.csv
outputs/results/0902-random_baseline_summary.csv
```

The random baseline has `spearman_rho = 0.009339` and `p_value = 0.834981` in the current output snapshot. This is effectively near zero, as expected for vectors with no lexical or semantic information. Treat it only as a lower-bound sanity check, not as a transformer model result.

Canonical synthesis files:

```text
outputs/tables/0701-main_results_for_thesis.csv
outputs/tables/0702-best_by_model_condition.csv
outputs/tables/0703-contextual_gain_summary.csv
outputs/tables/0704-pooling_layer_model_summary.csv
outputs/tables/0705-tokenization_fragmentation_summary.csv
outputs/tables/0706-triplet_probe_thesis_summary.csv
outputs/tables/0707-clean_vs_split_spearman_summary.csv
outputs/tables/0708-detailed_all_results_with_fragmentation.csv
outputs/figures/0701-main_results_overview.png
outputs/figures/0702-contextual_gain_by_model.png
outputs/figures/0703-triplet_probe_exploratory_summary.png
outputs/figures/0704-tokenization_fragmentation_overview.png
outputs/figures/0705-clean_vs_split_spearman.png
```

### Headline result

The strongest configuration currently present is:

| Result | Input | Model | Layer | Pooling | Spearman rho | p-value | Pairs |
|---|---|---|---:|---|---:|---:|---:|
| Best isolated | isolated | BERTurk | 1 | first | 0.4267 | 1.53e-23 | 500 |
| Best contextual | contextual | BERTurk | 7 | first | 0.5275 | 3.63e-37 | 500 |
| Global best | contextual | BERTurk | 7 | first | 0.5275 | 3.63e-37 | 500 |

Main interpretation:

> Turkish semantic similarity judgments are best approximated by contextual BERTurk embeddings from a middle transformer layer when multi-subtoken words are represented using first-subtoken pooling.

This is a moderate but clear correlation with human similarity rankings. It supports the project as a defensible thesis result, but it does not imply that model similarities fully reproduce human semantic judgments.

### Best configuration by model and input condition

| Input | Model | Layer | Pooling | Spearman rho | p-value |
|---|---|---:|---|---:|---:|
| isolated | BERTurk | 1 | first | 0.4267 | 1.53e-23 |
| isolated | mBERT | 12 | first | 0.1546 | 5.24e-04 |
| isolated | XLM-R | 1 | first | 0.1212 | 6.67e-03 |
| contextual | BERTurk | 7 | first | 0.5275 | 3.63e-37 |
| contextual | mBERT | 7 | last | 0.2441 | 3.22e-08 |
| contextual | XLM-R | 7 | first | 0.2478 | 1.97e-08 |

The Turkish-specific model is much stronger than the multilingual models in both isolated and contextual conditions. BERTurk is therefore the main model to foreground in the thesis narrative.

Tokenization split-pair rates:

| Model | Clean pairs | Split pairs | Both words split | Split pair percentage | Mean pair total subtokens |
|---|---:|---:|---:|---:|---:|
| BERTurk | 225 | 275 | 105 | 55.0% | 3.092 |
| mBERT | 12 | 488 | 402 | 97.6% | 5.496 |
| XLM-R | 93 | 407 | 219 | 81.4% | 4.030 |

Fragmentation is substantial for all models, but especially for mBERT and XLM-R. The project treats fragmentation as a descriptive and moderating variable, not as direct causal evidence.

### Pooling, layer, and model patterns

Pooling averages across models and layers:

| Input | Pooling | Mean Spearman rho |
|---|---|---:|
| isolated | first | 0.1972 |
| isolated | last | 0.1616 |
| isolated | mean | 0.1100 |
| isolated | max | 0.0884 |
| contextual | first | 0.2783 |
| contextual | last | 0.2200 |
| contextual | mean | 0.1732 |
| contextual | max | 0.0996 |

Layer averages across models and pooling strategies:

| Input | Layer | Mean Spearman rho |
|---|---:|---:|
| isolated | 1 | 0.1223 |
| isolated | 7 | 0.1224 |
| isolated | 12 | 0.1732 |
| contextual | 1 | 0.1057 |
| contextual | 7 | 0.2501 |
| contextual | 12 | 0.2225 |

Model averages across layers and pooling strategies:

| Input | Model | Mean Spearman rho | Max Spearman rho |
|---|---|---:|---:|
| isolated | BERTurk | 0.3235 | 0.4267 |
| isolated | mBERT | 0.0481 | 0.1546 |
| isolated | XLM-R | 0.0464 | 0.1212 |
| contextual | BERTurk | 0.3863 | 0.5275 |
| contextual | mBERT | 0.0874 | 0.2441 |
| contextual | XLM-R | 0.1045 | 0.2478 |

These tables show that the original expectation that mean pooling would dominate is not supported. The most defensible interpretation is instead that first-subtoken pooling is especially useful for Turkish word-level similarity in this setup, likely because the first subtoken often preserves root/stem information.

### Contextual gain

The contextual condition improves many configurations, especially around layer 7:

| Model | Layer | Pooling | Isolated rho | Contextual rho | Contextual minus isolated |
|---|---:|---|---:|---:|---:|
| BERTurk | 7 | first | 0.3151 | 0.5275 | 0.2125 |
| XLM-R | 7 | first | 0.0596 | 0.2478 | 0.1882 |
| mBERT | 7 | last | 0.0733 | 0.2441 | 0.1709 |
| mBERT | 7 | mean | 0.0341 | 0.1827 | 0.1486 |
| mBERT | 7 | first | 0.0904 | 0.2325 | 0.1421 |
| BERTurk | 7 | last | 0.2872 | 0.4139 | 0.1267 |
| XLM-R | 7 | last | 0.0676 | 0.1898 | 0.1222 |
| BERTurk | 7 | mean | 0.3051 | 0.4260 | 0.1209 |

This supports the claim that sentence-context averaging is useful for BERT-like models, especially for middle-layer representations.

### Fragmentation and clean-vs-split behavior

For the strongest BERTurk configurations, clean pairs correlate more strongly with human similarity than split pairs:

| Configuration | Clean-pair rho | Split-pair rho |
|---|---:|---:|
| isolated BERTurk layer 1 first | 0.5225 | 0.2416 |
| contextual BERTurk layer 7 first | 0.5756 | 0.4103 |

This pattern suggests that tokenization fragmentation can weaken alignment with human similarity judgments. However, the thesis should avoid claiming a simple causal effect, because split pairs may differ from clean pairs in word length, frequency, morphology, concreteness, and relation type.

The newest `0801` summaries refine this into mutually exclusive model-specific categories. For BERTurk, mean Spearman across all layer/pooling configurations decreases from clean pairs to one-word-split pairs to both-words-split pairs in both input conditions:

| Input | Clean mean rho | One word split mean rho | Both words split mean rho |
|---|---:|---:|---:|
| isolated | 0.4004 | 0.2060 | 0.0801 |
| contextual | 0.5371 | 0.3736 | 0.1066 |

This is useful thesis evidence for a fragmentation moderator pattern, while still requiring the causal caveat above.

Triplet probe headline results currently present:

| Robustness subset | Model | Layer | Pooling | Same-root sim | Same-suffix sim | Mean delta | Triplets |
|---|---|---:|---|---:|---:|---:|---:|
| all triplets | BERTurk | 1 | first | 0.7510 | 0.3811 | 0.3699 | 60 |
| all triplets | XLM-R | 1 | first | 0.9553 | 0.5684 | 0.3869 | 60 |
| all triplets | mBERT | 1 | first | 0.9793 | 0.3427 | 0.6366 | 60 |
| BERTurk all-split only | BERTurk | 1 | first | 0.9722 | 0.3416 | 0.6305 | 13 |
| BERTurk shared final subtoken | BERTurk | 1 | first | 0.8794 | 0.3965 | 0.4829 | 18 |

The triplet probe results are root-dominant in the current outputs: words sharing the same root are more similar than words sharing only the same suffix. The BERTurk all-split subset is small, so the probe should be described as exploratory robustness evidence.

These values are a snapshot of the checked-in/generated CSV outputs, not hard-coded conclusions. Re-run the notebooks after changing data, sentence generation, model settings, layers, or pooling strategies.

## Thesis Writing Handoff Summary

This section is intended for future thesis-writing sessions, including sessions in another LLM chat.

### What was done

1. The AnlamVer Turkish word-pair dataset was cleaned and used as the main intrinsic semantic similarity benchmark.
2. Unique AnlamVer words were tokenized with BERTurk, mBERT, and XLM-R to quantify fragmentation.
3. Isolated word embeddings were extracted for all three models across layers `1`, `7`, and `12`.
4. Four subtoken-to-word pooling strategies were compared: `first`, `last`, `mean`, and `max`.
5. Controlled generated Turkish context sentences were produced and validated for all target words.
6. Contextual word embeddings were extracted by locating target-word spans in sentence contexts and averaging sentence-level target embeddings.
7. Model cosine similarities were compared with AnlamVer human `Sim` ratings using Spearman correlation.
8. A synthetic morphological triplet probe tested whether embeddings are more influenced by shared roots or shared suffixes.
9. A synthesis notebook consolidated the results into thesis-ready `070x` tables and figures.
10. A final-results notebook added additive `0801` tables/figures, mutually exclusive clean/one-word/both-words fragmentation summaries, contextual-minus-isolated delta rankings, and triplet-probe caveat columns.
11. A random embedding baseline was added as a separate `090x` lower-bound sanity check. Its near-zero Spearman correlation confirms that random vector similarities do not systematically align with human AnlamVer similarity judgments.

### Hypothesis status

| Hypothesis | Status | Evidence |
|---|---|---|
| H1: Mean pooling should outperform first/last pooling. | Not supported. | First pooling is the best headline strategy in both isolated and contextual conditions. Mean pooling is competitive only in some BERTurk layer-12 settings. |
| H2: Fragmentation should hurt naive pooling. | Partially supported. | For BERTurk best configurations, clean-pair correlations are higher than split-pair correlations. However, this should be framed as descriptive/moderating evidence, not causal proof. |
| H3: Contextual sentence averaging should improve deeper/middle-layer representations. | Supported. | The largest gain is BERTurk layer 7 first pooling, from 0.3151 isolated to 0.5275 contextual. |
| H4: Middle layers should perform best. | Supported mainly for contextual embeddings. | Contextual layer 7 has the best average rho. Isolated results are less consistent and have a strong layer-1 BERTurk first-pooling result. |
| H5: Turkish-specific and multilingual models should differ. | Strongly supported. | BERTurk clearly outperforms mBERT and XLM-R in both isolated and contextual conditions. |
| Triplet probe expectation: shared roots should pull embeddings closer than shared suffixes if first pooling captures stem/root information. | Supported as exploratory evidence. | All headline triplet deltas are positive, including stricter robustness subsets. |

### Thesis-level interpretation

The original mean-pooling expectation is not supported. The stronger thesis claim is that, for Turkish AnlamVer semantic similarity, the first subtoken provides the most useful word-level representation in the tested models, especially in BERTurk. This result is compatible with a root/stem interpretation: in morphologically rich Turkish, the first subtoken often contains lexical root information, while later subtokens often encode suffixal or continuation material.

The best overall result comes from contextual BERTurk layer 7 first pooling. This supports a combined interpretation:

- Turkish-specific pretraining and tokenization matter.
- Contextual sentence averaging improves semantic alignment.
- Middle-layer contextual representations are especially useful.
- First-subtoken pooling is more effective than expected and may preserve root/stem information.

The triplet probe strengthens this interpretation but should not be overclaimed. It does not use AnlamVer human ratings, and it is synthetic and isolated. It is best described as a mechanistic follow-up that helps explain why first pooling works well in the main experiment.

### Suggested answer to the main research question

> Among the tested strategies, first-subtoken pooling produces the strongest alignment with human Turkish semantic similarity judgments, especially when applied to contextual BERTurk embeddings from layer 7. This suggests that word-level semantic information in this setup is not best recovered by averaging all subtokens; instead, root- or stem-aligned first subtokens appear to carry especially useful semantic information.

### Claims that are safe to make

- The best current model/condition/pooling setup is contextual BERTurk layer 7 first pooling.
- BERTurk substantially outperforms mBERT and XLM-R on this Turkish similarity task.
- Contextual sentence averaging improves the strongest BERTurk middle-layer result.
- First pooling is consistently strong and contradicts the initial expectation that mean pooling would dominate.
- Fragmentation is associated with weaker BERTurk clean-vs-split performance, but this should be treated cautiously.
- The triplet probe supports a root-dominant explanation for first pooling, but only as exploratory evidence.
- The random embedding baseline is close to zero and can be reported as a chance-level lower-bound sanity check when clearly separated from transformer results.

### Claims to avoid or qualify

- Do not claim that fragmentation causally lowers semantic performance without additional controls.
- Do not claim that first pooling is universally best for all Turkish NLP tasks.
- Do not claim that BERTurk fully captures human semantic similarity; the best rho is moderate, not near-perfect.
- Do not treat the triplet probe as the main evidence; the main evidence remains the AnlamVer Spearman analysis.
- Do not interpret generated context sentences as natural corpus data; they are controlled LLM-generated contexts.
- Do not mix the random embedding baseline into the main model comparison tables unless it is explicitly and visibly labelled as `random_baseline`.

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

### Random baseline

The random baseline assigns every unique AnlamVer word a fixed-seed random 768-dimensional vector and computes cosine similarity for the same 500 word pairs used in the main experiments. Because the vectors contain no lexical, morphological, or semantic information, the expected Spearman correlation with human `Sim` scores is close to zero.

In the current output snapshot, the random baseline gives `rho = 0.009339` with `p = 0.834981`. This supports its use as a chance-level lower-bound comparison. It should be reported separately from BERTurk, mBERT, and XLM-R results.

## Reproducibility

The main embedding notebooks set random seeds for Python, NumPy, and PyTorch where relevant. Model inference is deterministic enough for the intended analysis, but small numerical differences may occur across devices, PyTorch versions, and transformer backend settings.

The context sentence generation notebook is less deterministic because it calls an external LLM endpoint. To keep this step auditable, it saves raw JSONL responses and parsed validation outputs. Downstream contextual experiments should use the saved `data/processed/context_sentences_llm.csv` file for stable analysis.

The random baseline notebook uses `seed = 42` and writes only `0901` and `0902` outputs. It does not run transformer models and does not modify the `030x`, `050x`, `060x`, `070x`, or `080x` result files.

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
