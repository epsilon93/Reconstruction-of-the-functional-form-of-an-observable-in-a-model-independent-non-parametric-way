# Reconstruction-of-the-functional-form-of-an-observable-in-a-model-independent-non-parametric-way
# Astrophysical Data Analysis Pipelines

A scientific computing pipeline for eigenfunction/PCA-based spectral reconstruction, developed for astrophysics and observational data analysis.

---

## Repository Structure

```
.
├── pca_reconstruction/
│   ├── pcaSourceCode.py        # Core eigenfunction reconstruction engine
│   ├── run_eigen.py            # Wrapper for single-command pipeline execution
│   ├── generate_prior.py       # Prior patch file generator
│   └── params.in               # Centralized parameter configuration
├── data/
│   └── ...                     # Input data files
├── outputs/
│   └── ...                     # Reconstructed spectra, posterior samples, plots
├── environment.yml             # Conda environment specification
└── README.md
```

---

## Pipeline Overview

### Eigenfunction / PCA Reconstruction Pipeline

A Python port of a Fortran eigenfunction reconstruction program. Implements chi-squared minimisation, covariance matrix construction, eigen-decomposition, and multi-truncation-level spectral reconstruction.

**Key features:**
- Core reconstruction engine (`pcaSourceCode.py`) ported from Fortran with verified logic
- Single-command execution via `run_eigen.py` + `params.in` parameter file
- `generate_prior.py` constructs prior patch files parametrised by zero-position and step size, ensuring each patch straddles zero
- Optional CLI flags: blank values in `params.in` fall back to script defaults

**Usage:**

```bash
# Edit parameters as needed
nano params.in

# Run full pipeline
python run_eigen.py

# Generate a prior patch file
python generate_prior.py --zero-pos <value> --step <value>
```

**Output:** Reconstructed spectra at multiple truncation levels, covariance diagnostics, and eigenvalue spectra.

---

## Environment Setup

This project targets **macOS Apple Silicon (ARM)** with Python 3.13 and PyTensor 2.30.3. The steps below also work on Linux x86-64.

### Create the Conda environment

```bash
conda env create -f environment.yml
conda activate pymc_env
```

### Manual setup (if not using `environment.yml`)

```bash
conda create -n pymc_env python=3.13
conda activate pymc_env
pip install numpy scipy matplotlib
```

---

## Dependencies

| Package     | Role                                      |
|-------------|-------------------------------------------|
| NumPy       | Numerical arrays and linear algebra       |
| SciPy       | Numerical integration utilities           |
| Matplotlib  | Plotting and diagnostics                  |

---

## Data

Input data files (prior files, observational spectra) are **not** included in this repository due to size and licensing constraints. Place them in the `data/` directory before running either pipeline. See individual script headers for expected file formats.

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
