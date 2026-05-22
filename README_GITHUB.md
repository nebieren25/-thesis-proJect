# Subword Tokenization and Turkish Semantic Similarity

This repository contains the code component of a BSc thesis project on Turkish subword tokenization and word-level semantic similarity. The project evaluates how different methods for combining subtoken embeddings into a single word embedding align with human semantic similarity judgments from the AnlamVer dataset.

The analysis is notebook-based so that each stage of the experimental pipeline can be inspected, rerun, and audited independently.

## Project Aim

Transformer language models usually split words into subword tokens. This is especially important for Turkish, where agglutinative morphology can produce words made from a stem plus several suffixes. The central question of this project is:

> Which subtoken-to-word pooling strategy produces word embeddings that best match human semantic similarity judgments for Turkish?

The project compares model-derived cosine similarities against human AnlamVer similarity scores using Spearman rank correlation.

## Main Experiments

The repository covers:

- tokenization analysis of AnlamVer word pairs,
- isolated-word embedding experiments,
- contextual sentence-averaged embedding experiments,
- model and layer comparisons,
- tokenization fragmentation analysis,
- a Turkish morphology triplet probe,
- a random-vector baseline,
- bootstrap robustness tests for pooling strategies.

The main pooling strategies are:

- `first`: first real subtoken embedding,
- `last`: last real subtoken embedding,
- `mean`: average of all real subtoken embeddings,
- `max`: element-wise maximum across real subtoken embeddings.

Special tokens such as `[CLS]`, `[SEP]`, `<s>`, and `</s>` are excluded from pooling.

## Repository Structure

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
|-- 10_pooling_bootstrap_tests.ipynb
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|-- outputs/
|   |-- figures/
|   |-- results/
|   `-- tables/
|-- triplet_files/
|-- requirements.txt
`-- README_GITHUB.md
```

## Notebook Order

Run the notebooks from the repository root so that relative paths resolve correctly.

| Notebook | Purpose |
|---|---|
| `01_tokenization_analysis.ipynb` | Cleans AnlamVer and measures tokenization fragmentation. |
| `02_isolated_embedding_experiment.ipynb` | Runs the initial BERTurk isolated-word experiment. |
| `03_all_model_isolated_embedding_experiment.ipynb` | Extends isolated embeddings to BERTurk, mBERT, and XLM-R. |
| `04_generate_llm_context_sentences.ipynb` | Generates context sentences for contextual experiments. |
| `05_contextual_embedding_experiment.ipynb` | Extracts contextual sentence-averaged word embeddings. |
| `06_triplet_probe_experiment.ipynb` | Tests whether representations are more root- or suffix-dominant. |
| `07_results_synthesis_analysis.ipynb` | Consolidates result tables and figures. |
| `08_final_results_figures.ipynb` | Produces final compact thesis tables and plots. |
| `09_random_embedding_baseline.ipynb` | Adds a random embedding baseline. |
| `10_pooling_bootstrap_tests.ipynb` | Runs bootstrap tests for the headline pooling comparison. |

For reviewing final results, start with notebooks `07`, `08`, `09`, and `10`. The earlier notebooks document the data preparation and model inference stages.

## Installation

Create a fresh Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name thesis-subword-tokenization
```

The project was developed with Python notebooks and the dependencies listed in `requirements.txt`.

Transformer model weights may be downloaded from Hugging Face the first time the model notebooks are run.

## Data

The main input dataset is:

```text
data/raw/anlamver-final.csv
```

The project uses the AnlamVer Turkish semantic similarity dataset. The main evaluation column is the human semantic similarity score, used as the gold standard for comparison with model-derived cosine similarities.

Generated context sentences and API response logs are stored under:

```text
data/processed/
```

This allows the contextual experiments to be inspected without rerunning the context-generation step.

## API Key

Notebook `04_generate_llm_context_sentences.ipynb` can call an OpenAI-compatible API endpoint through the `openai` Python SDK. To rerun that notebook, provide the key through the environment or a local `.env` file:

```text
OPENCODE_API_KEY=your_key_here
```

Do not commit a real `.env` file or private API keys. Use `.env.example` as the template.

## Outputs

Generated result tables and figures are written to:

```text
outputs/
```

Important outputs include:

```text
outputs/tables/0701-main_results_for_thesis.csv
outputs/tables/0702-best_by_model_condition.csv
outputs/tables/0705-tokenization_fragmentation_summary.csv
outputs/tables/0801-best_overall_configurations.csv
outputs/tables/0801-best_isolated_vs_contextual_configurations.csv
outputs/tables/0801-clean_one_both_spearman_summary.csv
outputs/tables/0801-triplet_probe_summary_with_caveats.csv
outputs/tables/1001-pooling_bootstrap_tests.csv
outputs/figures/0701-main_results_overview.png
outputs/figures/0801-best_spearman_by_model_condition.png
outputs/figures/0801-spearman_by_fragmentation_category.png
outputs/figures/0801-triplet_same_root_vs_same_suffix_summary.png
```

Generated outputs are analysis artifacts, not hard-coded conclusions. If the dataset, models, context sentences, layers, or pooling strategies change, rerun the relevant notebooks and regenerate downstream outputs.

## Current Result Snapshot

In the current workspace outputs, contextual BERTurk at layer `7` with `first` pooling is identified as the strongest configuration. The bootstrap analysis in `10_pooling_bootstrap_tests.ipynb` supports `first` pooling over `mean`, `last`, and `max` for that headline configuration.

This result should be interpreted as the outcome of the current pipeline and data state, not as a permanent claim independent of rerunning the experiments.

## Reproducibility Notes

- Run notebooks from the repository root.
- Keep generated model caches outside the repository when possible.
- Do not commit `.venv/`, `.env`, Hugging Face caches, or private API keys.
- Large generated outputs can be regenerated from the notebook pipeline.
- The synthesis and plotting notebooks are faster to rerun than the transformer inference notebooks.

## Thesis Context

This project supports the thesis:

```text
Where Does Meaning Live in Subword Tokenization?
Evaluating Subtoken-to-Word Embedding Composition for Turkish on AnlamVer
```

Programme: BSc Cognitive Science & Artificial Intelligence, Tilburg University.
