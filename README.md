# Asia Causal Discovery Reproduction

This repository is a minimal reproducibility package for the paper experiments on the **Asia** Bayesian network using a **metadata-only base prompt**.

## Background

Causal inference aims to distinguish correlation from direct causal influence. In this project, large language models are asked to estimate whether a directed edge `A -> B` exists between variables in a known causal graph, using only dataset metadata. To make the evaluation more realistic, we do not rely only on edge accuracy; we also include an uncertainty-aware evaluation scheme controlled by the penalty parameters `alpha` and `beta`.

## What This Package Reproduces

This package reproduces the paper's **Asia dataset** experiments for the:

- metadata-only setting
- vanilla base prompt
- model list defined in `config.yaml`
- uncertainty-aware evaluation over all `(alpha, beta)` combinations

The package is intentionally small and excludes unrelated datasets, prompt variants, plotting code, and larger project machinery.

## Included Files

### Data

`DataFolder/asia/`

- `asia_description.json`
- `asia_graph.json`
- `MetaData/clean/asia_description.json`
- `MetaData/noisy/asia_L1_names_only.json`
- `MetaData/noisy/asia_L2_labels_levels.json`
- `MetaData/noisy/asia_L3_role_description.json`

### Code

- `main.py`
- `client.py`
- `parser.py`
- `prompt_loader.py`
- `utils.py`

### Config

- `config.yaml`
- `.env.example`
- `requirements.txt`

## Configuration

The experiment is controlled through `config.yaml`. The current package uses:

- dataset: `./DataFolder/asia`
- temperature: `0.0`
- threshold: `0.7`
- `alpha, beta in {0.0, 0.1, ..., 1.0}`

## How Reproduction Works

When you run `main.py`, the pipeline does the following:

1. Loads the Asia dataset description and ground-truth graph.
2. Discovers the available metadata variants in `DataFolder/asia/MetaData/`.
3. For each metadata file and each model, queries the model twice:
   - once with the **edge** prompt
   - once with the **no-edge** prompt
4. Stores all raw model outputs in a single file:
   - `outputs/raw_llm_results.json`
5. Immediately evaluates those outputs against the ground-truth graph.
6. Computes uncertainty-aware metrics across the full `(alpha, beta)` grid.
7. Stores the evaluation results in a single file:
   - `outputs/evaluation_results.csv`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file:

```bash
cp .env.example .env
```

4. Add your OpenRouter API key to `.env`.

## Run

```bash
python main.py
```

or

```bash
bash run.sh
```

## Outputs

After a run, only two main output files are produced:

- `outputs/raw_llm_results.json`
- `outputs/evaluation_results.csv`

## Notes

- This package is restricted to the paper's base-prompt path to keep it lightweight and easy to upload.
- The code is still flexible enough to extend to another dataset by adding a new dataset folder and updating `config.yaml`.
