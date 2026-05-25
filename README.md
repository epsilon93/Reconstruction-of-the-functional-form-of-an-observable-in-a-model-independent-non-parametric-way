# Astrophysical Data Analysis Pipelines

A scientific computing pipeline for eigenfunction/PCA-based spectral reconstruction, developed for astrophysics and observational data analysis.

## Description

This repository provides a modular, reproducible Python pipeline for eigenfunction decomposition and PCA-based spectral reconstruction of astrophysical observational data. The core engine (`pcaSourceCode.py`) is a rigorously verified port of a legacy Fortran program, implementing covariance matrix construction from input spectra, full eigen-decomposition, chi-squared minimisation against a prior, and reconstruction at multiple user-specified truncation levels. A lightweight wrapper (`run_eigen.py`) driven by a single parameter file (`params.in`) enables end-to-end execution with a single command, while `generate_prior.py` automates the construction of prior patch files parametrised by zero-position and step size. The pipeline is designed for reproducibility, modularity, and straightforward extension to new datasets or reconstruction schemes.

Theory and the principle are explained in the following papers:

~Sharma Ranbir, Mukherjee Ankan, Jassal H K (https://arxiv.org/abs/2004.01393)
~Sharma Ranbir, Jassal H K(https://arxiv.org/abs/2211.13608)


---

## Repository Structure

```
.
├── pcaSourceCode.py        # Stage 1 – core eigenfunction reconstruction engine
├── pymc_main.py            # Stage 2 – PyMC MCMC inference + corner plot
├── generate_prior.py       # Stage 0 – prior patch file generator
├── run_eigen.py            # Single-command wrapper (reads params.in)
├── params.in               # Centralised parameter configuration
├── environment.yml         # Conda environment specification
├── inputFolder/
│   ├── data_file.dat       # Observational data  (r, value, error)
│   └── input.dat           # Prior patch file    (r1_upper, step_size)
└── outputFiles/            # All generated outputs (created automatically)
```

---

## Pipeline Overview

### Eigenfunction / PCA Reconstruction Pipeline

A Python port of a Fortran eigenfunction reconstruction program. Implements chi-squared minimisation, covariance matrix construction, eigen-decomposition, and multi-truncation-level spectral reconstruction, followed by full Bayesian MCMC inference with PyMC.

**Stage 0 — Prior generation (`generate_prior.py`, optional)**

Constructs the `input.dat` prior patch file. Each patch is a `(r1_upper, step_size)` pair whose candidate grid straddles zero.

```bash
python generate_prior.py --output inputFolder/input.dat
```

**Stage 1 — PCA / eigenfunction reconstruction (`pcaSourceCode.py`)**

Grid-searches chi-squared on all prior patches, builds the covariance matrix, eigen-decomposes it, and reconstructs the observable at truncation levels `P-5` to `P`.

```bash
python pcaSourceCode.py \
    --input-dir  inputFolder \
    --output-dir outputFiles \
    --tag        my_run
```

**Stage 2 — Bayesian MCMC inference (`pymc_main.py`)**

Reads Stage-1 outputs, builds a PyMC model, and samples the posterior. Produces a corner plot (PNG) and a trace pickle. The sampler is selectable via `--sampler`:

| Flag | Backend | Notes |
|---|---|---|
| `nuts` (default) | PyTensor | Standard NUTS |
| `hmc` | PyTensor | Hamiltonian MC |
| `metropolis` | PyTensor | Gradient-free |
| `nuts-numpyro` | JAX | Fast on Apple Silicon; requires `numpyro` + `jax` |
| `nuts-blackjax` | JAX | Fast on Apple Silicon; requires `blackjax` + `jax` |

```bash
python pymc_main.py \
    --data-file     inputFolder/data_file.dat \
    --data-ini-file outputFiles/DATA_my_run.txt \
    --coeff-file    outputFiles/resultant_coff_my_run_h0_7.txt \
    --sampler       nuts
```

**Single-command execution via `run_eigen.py`**

All three stages are driven by `params.in` through a single wrapper:

```bash
# Edit parameters as needed, then run
python run_eigen.py            # reads ./params.in
python run_eigen.py my_run.in  # or an alternate parameter file
```

`params.in` controls which stages run (`stages = 1`, `stages = 2`, or `stages = 1,2`), whether to auto-generate the prior (`generate_prior = true`), and every CLI flag for all three scripts. Blank values fall back to each script's own defaults.

**Outputs** (all written to `--output-dir`):

| File pattern | Content |
|---|---|
| `DATA_<tag>.txt` | Per-patch chi² and best-fit coefficients (Stage 1) |
| `COV_MATRIX_<tag>.txt` | Coefficient covariance matrix |
| `EIGENVALUES_<tag>.txt` / `EIGENVECTORS_<tag>.txt` | Eigen-decomposition |
| `EIGENFUNCTIONS_<tag>.txt` / `NEIGENFNS_<tag>.txt` | Eigenfunctions on data grid |
| `DATA_FINAL_N_<tag>.txt` | Stage-2 grid search results |
| `resultant_coff_<tag>_h0_<k>.txt` | Reconstruction coefficients at truncation level k |
| `resultant_<tag>_h0_<k>.txt` | Reconstructed observable at truncation level k |
| `reconstruction_<tag>.png` | Overlay plot of all truncation levels vs data |
| `pymc/pymc_pairplot.png` | Corner plot of posterior (Stage 2) |
| `pymc/pymc_trace.pkl` | Serialised posterior trace (Stage 2) |

---

## Environment Setup

The project is tested on **macOS Apple Silicon (ARM)** and **Linux x86-64** with Python 3.11.

### Create the Conda environment

```bash
conda env create -f environment.yml
conda activate pca-recon
```

### Update an existing environment

```bash
conda env update -f environment.yml --prune
```

### Optional JAX samplers (faster on Apple Silicon)

The PyTensor C-backend is disabled on Apple Silicon (`pytensor.config.cxx = ""`), making the JAX-backed NUTS samplers the recommended choice on that hardware. To enable them, uncomment the relevant block in `environment.yml` before creating the environment, or install manually:

```bash
# NumPyro backend  (--sampler nuts-numpyro)
pip install "jax[cpu]" numpyro

# BlackJAX backend  (--sampler nuts-blackjax)
pip install "jax[cpu]" blackjax
```

> **Apple Silicon GPU/Metal**: replace `jax[cpu]` with `jax-metal` for hardware-accelerated JAX.

---

## Dependencies

All core packages are pinned in `environment.yml` and installed from `conda-forge`.

| Package | Version | Role |
|---|---|---|
| Python | 3.11 | Runtime |
| NumPy | ≥1.24, <2.0 | Arrays, linear algebra, file I/O |
| SciPy | ≥1.10 | Numerical integration (`quad`), spline interpolation |
| tqdm | ≥4.65 | Progress bars during Stage-1 grid search |
| PyMC | ≥5.0 | Bayesian model definition and sampling (Stage 2) |
| PyTensor | ≥2.18 | Symbolic tensor backend for PyMC |
| ArviZ | ≥0.17 | Posterior diagnostics, `az.summary`, `az.hdi`, corner plots |
| SymPy | ≥1.12 | Symbolic error-function construction in Stage 2 |
| Matplotlib | ≥3.7 | Reconstruction overlay plots and corner plots |

**Optional** (JAX samplers — not installed by default):

| Package | Role |
|---|---|
| jax / jaxlib | XLA-based JIT compiler required by both JAX backends |
| numpyro | `--sampler nuts-numpyro` |
| blackjax | `--sampler nuts-blackjax` |

---

## Data

Input data files (prior files, observational spectra) are **not** included in this repository due to size and licensing constraints. Place them in the `inputFolder/` directory before running the pipeline. See individual script headers for expected file formats (`data_file.dat`: three columns `r, value, error`; `input.dat`: two columns `r1_upper, step_size`).

---

## Known Limitations & Open Questions

- The prior file generation logic in `generate_prior.py` may not exactly match the original Fortran `input.dat` generator. If the original `input.dat` is located, `generate_prior.py` should be updated to replicate its enumeration pattern precisely — the Fortran source contains non-standard cumulative counting logic that does not correspond to standard enumeration.
- Bit-for-bit verification of the Fortran-to-Python translation across multiple parameter combinations is recommended before treating any ported module as production-ready.

---

## Contributing

This is a research codebase under active development. If you find a bug or discrepancy against the original Fortran behaviour, please open an issue with a minimal reproducible example and the parameter set used.

---

## License

MIT License. See `LICENSE` for details.
