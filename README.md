# Machine Learning for Chemistry: Molecular Property Prediction with GNNs

This repository contains a comprehensive framework for predicting molecular properties using Graph Neural Networks (GNNs). The primary focus is on aqueous solubility prediction—a critical task in drug discovery and chemical engineering—leveraging the Delaney (ESOL) dataset. The project implements a rigorous, multi-stage Hyperparameter Optimization (HPO) pipeline to systematically identify optimal model architectures and training configurations.

## Table of Contents
1. [Project Description](#project-description)
2. [Dataset: ESOL](#dataset-esol)
3. [Methodology](#methodology)
   - [Model Architectures](#model-architectures)
   - [Two-Stage HPO Pipeline](#two-stage-hpo-pipeline)
4. [Project Structure](#project-structure)
5. [Installation and Setup](#installation-and-setup)
6. [Entrypoint](#entrypoint)
7. [Quickstart](#quickstart)
8. [Usage](#usage)
   - [Compute Environment](#compute-environment)
   - [Run HPO](#run-hpo)
   - [Evaluate Winner Robustness](#evaluate-winner-robustness)
   - [Notebook Analysis Workflow](#notebook-analysis-workflow)
9. [Outputs](#outputs)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

---

## Project Description
Molecular solubility is a fundamental property that determines the bioavailability and efficacy of chemical compounds. Traditional experimental methods for determining solubility are time-consuming and resource-intensive. This project explores the use of Graph Neural Networks to learn spatial and topological features directly from molecular graphs, enabling fast and accurate property prediction.

Beyond simple prediction, this work focuses on the "science of HPO." By comparing different sampling strategies (Bayesian Optimization, Tree-structured Parzen Estimator, and Random Search) across a high-dimensional search space, we aim to understand which architectural choices (e.g., message-passing operators, pooling strategies) truly drive performance on small-molecule benchmarks.

## Dataset: ESOL
The Delaney (ESOL) dataset is a standard benchmark in the MoleculeNet suite. It consists of 1,128 compounds with associated water solubility data.
- **Input:** SMILES strings, which are converted into molecular graphs where atoms are nodes and bonds are edges.
- **Target:** Log solubility (mol/L).
- **Challenges:** The dataset is relatively small, making it prone to overfitting and sensitive to hyperparameter choices, which necessitates the robust regularization and HPO strategies implemented here.

---

## Methodology

### Model Architectures
The framework supports several state-of-the-art GNN layers, allowing for a comparative study of inductive biases:
- **GIN (Graph Isomorphism Network):** Utilizes a sum-aggregation strategy proven to be as powerful as the Weisfeiler-Lehman graph isomorphism test.
- **GAT (Graph Attention Network):** Employs multi-head attention mechanisms to weigh the importance of neighboring atoms differently.
- **GraphSAGE:** Uses sampling and various aggregation functions (mean, max, etc.) to learn neighborhood embeddings.
- **GCN (Graph Convolutional Network):** The standard spectral-based baseline for graph learning.

### Two-Stage HPO Pipeline
To navigate the complex search space efficiently, the project employs a two-stage approach:

#### 1. Coarse Pass (Exploration)
The coarse pass is designed to identify "productive regions" of the search space. We run 150 trials across 15 different seeds for each sampler. This stage tests broad hypotheses regarding:
- **Layer Expressivity:** Comparing GIN + Sum Pooling against other combinations.
- **Optimization:** Evaluating Adam, AdamW, and RMSprop against a baseline of SGD.
- **Regularization:** Testing the impact of Dropout and Weight Decay on generalization for the small ESOL dataset.
- **Sampler Performance:** A head-to-head comparison of Bayesian Optimization (using BoTorch and qNEI) versus TPE and Random Search.

#### 2. Fine Pass (Exploitation)
Based on the results of the coarse pass (analyzed via fANOVA importance and contour plots), the fine pass narrows the priors around the winning configurations. This stage focuses on precision tuning of the most sensitive parameters to squeeze out final performance gains.

---

## Project Structure
```
machine_learning_for_chemistry/
├── notebooks/                # Documentation and Analysis
│   ├── main.ipynb            # Final presentation of results and narrative
│   └── analysis/             # HPO diagnostics and visualization scripts
│
├── src/                      # Core Implementation
│   ├── run_bayes.py          # Bayesian Optimization search runner
│   ├── run_optuna.py         # TPE and Random Search runner
│   ├── helpers/              
│   │   ├── models.py         # PyTorch Geometric GNN definitions
│   │   ├── training.py       # Training loops and early stopping logic
│   │   └── search_shared.py  # Shared HPO utilities
│   └── evaluate_...          # Scripts for evaluating final model performance
│
├── scripts/                  # Orchestration
│   ├── coarse/               # Scripts for the initial exploration phase
│   ├── fine/                 # Scripts for the precision tuning phase
│   └── benchmark/            # Unit tests and performance benchmarks
│
├── output/                   # Persistent Storage
│   ├── simulation/           # Raw HPO trial data (CSV/JSON) <must be downloaded>
│   ├── test_eval/            # Post simulation tests results
│   └── figures/              # Generated plots and visualizations
│
├── Makefile                  # Automation for setup and data retrieval
├── pyproject.toml            # Python package configuration
└── README.md                 # Project documentation
```

---

## Installation and Setup

### Requirements
- Python 3.11 or higher (matches `pyproject.toml`).
- [`uv`](https://docs.astral.sh/uv/) for environment + dependency management.
- Linux/macOS shell environment for the provided `scripts/*.sh` wrappers.
- Chrome/Chromium if you export notebook figures with Plotly/Kaleido.

> Project convention: run scripts with `uv run ...` rather than plain `python ...`.

### Prerequisites
See the Requirements section above (Python/uv/Chrome) and the Python dependency list from `pyproject.toml`.


### Recommended Installation with `uv`
```bash
uv sync
```
Run commands with:
```bash
uv run python <script>.py ...
```


### Simulation Data Retrieval
The HPO simulation data is large and stored externally. To download the pre-computed results for analysis:
```bash
make sim-results
```
This command retrieves the zip archive from the project's SharePoint storage and extracts it into the `output/simulation/` directory.
Alternatively download and extract/rename from here
```bash
https://amsuni-my.sharepoint.com/personal/salome_poulain_student_uva_nl/_layouts/15/download.aspx?share=IQBObImgeBweT7w5nSuRrFF0AUjBm5Ch43jPvHMVDIslOIY
```

### ESOL Dataset Retrieval
The Simulation Dataset will be retrieved in the code that needs it, it does so using the torch module:
```
from torch_geometric.datasets import MoleculeNet
```


### Automation with Makefile
A `Makefile` is provided to automate common tasks:

- **Download Simulation Data:**
  ```bash
  make sim-results
  ```
- **Cleanup Caches and Build Artifacts:**
  Remove Python `__pycache__`, `.mypy_cache`, `.ruff_cache`, and other temporary files:
  ```bash
  make clean
  ```

---

## Entrypoint

```bash
notebooks/main.ipynb
```

This notebooks does not run simulations but is the report and overview of all the projects findings. 


## Quickstart

Run one small random-search smoke test:

```bash
uv run python src/run_random.py \
  --search-space coarse \
  --trials 3 \
  --search-epochs 5 \
  --output-dir output/simulation \
  --run-name smoke_random_3x5
```

You can inspect generated artifacts in:

`output/simulation/smoke_random_3x5/`

---

## Usage

### Compute Environment
The large-scale simulations in this project were performed on [Snellius](https://www.surf.nl/en/snellius-the-national-supercomputer). The HPO passes are designed to be orchestrated via SLURM for parallel execution across multiple nodes.

### Run HPO

You can run search drivers directly:

```bash
# BoTorch
uv run python src/run_bayes.py --help

# Optuna (TPE / random / GP / CMA-ES)
uv run python src/run_optuna.py --help

# Random search
uv run python src/run_random.py --help
```

For SLURM-based execution, submit scripts from `scripts/`:
```bash
sbatch scripts/coarse/optuna_tpe.sh
```

### Evaluate Winner Robustness

Use fresh train/val/test splits to estimate validation-to-test generalization gap:

```bash
uv run python src/evaluate_hpo_winner_fresh_splits.py \
  --config-json output/figures/pass_comparison/fine_best_config_for_test_eval.json \
  --n-splits 5 \
  --output-csv output/test_eval/bo_winner_fresh_splits.csv
```


This script saves tabular/JSON outputs only (no plotting).

### Notebook Analysis Workflow

1. Run standalone scripts in `src/` to generate artifacts.
2. Open analysis notebooks in `notebooks/analysis/`.
3. Point notebook paths (e.g., `TOPK_OUTPUT_DIR`) to the produced run folder.
4. Use notebooks only for loading, plotting, and interpretation.

---

## Outputs

Typical output folders:

- `output/simulation/<run_name>/`
  - `trials.sqlite`
  - `trials.csv`
  - `trials.json`
  - `metadata.json`

- `output/topk_learning_curves/<run_name>/`
  - `topk_selected.csv`
  - `topk_repro.csv`
  - `topk_curves_long.csv`
  - `topk_curves.json`
  - `topk_manifest.json`

- `output/test_eval/`
  - fresh-split winner evaluation CSVs.

---

## Troubleshooting

- **Jupyter Notebooks Not Working:** Ensure the virtual environment is activated before launching Jupyter:
  ```bash
  source .venv/bin/activate
  jupyter lab
  ```
  Or run via `uv`:
  ```bash
  uv run jupyter lab
  ```
  If Jupyter is not found, reinstall dependencies: `uv sync`.

- **Notebooks Not Finding Modules:** Jupyter must run inside the activated virtual environment. If notebooks can't import project modules, restart the kernel and check that the notebook is using the correct Python interpreter (should point to `.venv/bin/python`).

- **VS Code Path Resolution:** This project is optimized for development in Visual Studio Code. To ensure that relative imports and data paths resolve correctly, you must open the `machine_learning_for_chemistry/` directory as your primary workspace root. Ensure your `.vscode/settings.json` is configured to use the project's virtual environment.
- **`python` works but scripts fail with missing modules:** Prefer `uv run python ...` so dependency resolution matches `pyproject.toml`.
- **Missing Plotly Images:** If the analysis notebooks fail to export plots, ensure that Chrome is installed. You may also need to run `.venv/bin/kaleido_get_chrome` to ensure the headless browser is correctly configured.
- **ModuleNotFoundError:** Ensure the virtual environment is activated and the project was installed using `uv sync`.
- **HPC/SLURM Issues:** If running on Snellius, ensure your account has the necessary quotas and that the `conda` or `venv` environments are correctly loaded in your SLURM submission scripts.

---

## License
This project is licensed under the terms specified in the `LICENSE` file.
