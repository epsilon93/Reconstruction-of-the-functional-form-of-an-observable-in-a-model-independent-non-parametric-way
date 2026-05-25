#!/usr/bin/env python3
"""
Python port of the Fortran eigenfunction-reconstruction program.

Pipeline (mirrors the original):
    1. Stage-1 grid search (chi^2 minimisation on every "patch" of parameter
       space) using the basis  funct(r)^j , j = 0..P-1.
    2. Build the covariance matrix of the per-patch best-fit coefficients.
    3. Eigen-decompose the covariance matrix (DSYEV equivalent) and sort
       eigen-pairs by ASCENDING ABSOLUTE EIGENVALUE.
    4. Build (and normalise) the eigenfunctions on the data grid.
    5. Stage-2 grid search using the eigenfunctions as the basis.
    6. Pick the global minimum across patches; reconstruct the function for
       k = P-5, P-4, ..., P modes and write coefficients + recon files.

Input file conventions (whitespace-separated, like Fortran list-directed read):
    --data-file    rows of  r , value , error          (data_points rows)
    --prior-file   rows of  r1_upper , step_size       (total_rows rows)

All outputs are written into --output-dir (created automatically).
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Basis function (the Fortran `funct`).  Edit here if you want a different
# parametrisation -- everything downstream uses this transparently.
# --------------------------------------------------------------------------- #
def funct(t: np.ndarray) -> np.ndarray:
    """funct(t) = t / (1 + t)."""
    return t / (1.0 + t)


# --------------------------------------------------------------------------- #
# Per-patch worker.  Module-level globals are populated by the Pool initializer
# so the (small) shared arrays are not re-pickled on every call.
# --------------------------------------------------------------------------- #
_BASIS = None       # (D, P)  design matrix used inside chi^2
_HS = None          # (D,)    measurements
_ERR = None         # (D,)    errors (1-sigma)
_NUM_PARAM = None
_L100 = None
_CHUNK = None
# pre-computed quadratic form pieces (depend only on basis/HS/ERR)
_A = None           # (P, P)  X^T W X
_V = None           # (P,)    X^T W y
_CONST = None       # scalar  y^T W y


def _init_worker(basis, hs, err, num_param, l100, chunk):
    global _BASIS, _HS, _ERR, _NUM_PARAM, _L100, _CHUNK, _A, _V, _CONST
    _BASIS = basis
    _HS = hs
    _ERR = err
    _NUM_PARAM = num_param
    _L100 = l100
    _CHUNK = chunk
    w = 1.0 / (err ** 2)
    Xw = basis * w[:, None]
    _A = basis.T @ Xw                          # (P, P)
    _V = Xw.T @ hs                             # (P,)
    _CONST = float(np.sum(hs * hs * w))        # y^T W y


def _patch_search(patch):
    """Search the l100**P grid on one patch; return (chi2_min, best_coeffs)."""
    r1, dp = patch

    # b(i) = r1 - i*dp for i = 1..l100   (Fortran convention preserved)
    b_grid = r1 - np.arange(1, _L100 + 1, dtype=np.float64) * dp

    n_total = _L100 ** _NUM_PARAM
    best_chi = np.inf
    best_coeff = np.zeros(_NUM_PARAM)

    # Stream the cartesian product in chunks so memory stays bounded.
    for start in range(0, n_total, _CHUNK):
        end = min(start + _CHUNK, n_total)
        flat = np.arange(start, end, dtype=np.int64)

        # Decode flat index -> multi-index in base l100.
        # Digit 0 is the least significant, matching Fortran's K(1).
        multi = np.empty((end - start, _NUM_PARAM), dtype=np.int64)
        rem = flat
        for j in range(_NUM_PARAM):
            multi[:, j] = rem % _L100
            rem //= _L100

        coeffs = b_grid[multi]                                  # (chunk, P)

        # chi^2(c) = c.A.c - 2 v.c + const
        quad = np.einsum("ij,jk,ik->i", coeffs, _A, coeffs)
        lin = coeffs @ _V
        chi2 = quad - 2.0 * lin + _CONST

        k = int(np.argmin(chi2))
        if chi2[k] < best_chi:
            best_chi = float(chi2[k])
            best_coeff = coeffs[k].copy()

    return best_chi, best_coeff


# --------------------------------------------------------------------------- #
# Plotting: overlay all truncated reconstructions on the data (with error bars)
# --------------------------------------------------------------------------- #
def plot_results(r_arr, hs_arr, err_arr, coeffs_full, ks, tag, out_dir,
                 n_dense=400, show=False):
    """Overlay reconstruction curves (k = ks[0]..ks[-1]) on the data points.

    Curves are evaluated on a dense r-grid (so they look smooth), while the
    data points are drawn as error-bar markers.  One PNG is written into
    out_dir and its path is returned.
    """
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")          # headless-safe default
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available -- skipping plot.")
        return None

    # Smooth grid spanning the data range.
    r_dense = np.linspace(r_arr.min(), r_arr.max(), n_dense)
    # Design matrix on the dense grid: funct(r_dense)^j for j = 0..P-1
    P = len(coeffs_full)
    dense_basis = np.vander(funct(r_dense), P, increasing=True)  # (n_dense, P)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Data with error bars.
    # sort for clean connectors (does not matter for errorbar but is tidy)
    order = np.argsort(r_arr)
    ax.errorbar(r_arr[order], hs_arr[order], yerr=err_arr[order],
                fmt="o", ms=4, capsize=3, color="black",
                ecolor="gray", elinewidth=0.8, alpha=0.85,
                label="data", zorder=3)

    # One curve per truncation level.
    cmap = plt.get_cmap("viridis")
    ks = sorted(k for k in ks if 1 <= k <= P)
    for idx, k in enumerate(ks):
        frac = idx / max(1, len(ks) - 1)
        color = cmap(0.15 + 0.75 * frac)
        coff_k = coeffs_full[:k]
        curve = dense_basis[:, :k] @ coff_k
        ax.plot(r_dense, curve, lw=1.8, color=color,
                label=f"k = {k}", zorder=4)

    ax.set_xlabel("r")
    ax.set_ylabel("value")
    ax.set_title(f"Reconstruction vs. data  —  tag: {tag}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()

    out_path = Path(out_dir) / f"reconstruction_{tag}.png"
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"[plot] saved {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="./input_dir",
                   help="Folder containing the input files (default: ./inputFolder).")
    p.add_argument("--data-file", default="data_file.dat",
                   help="Measurement file (3 columns: r, value, error). "
                        "Resolved against --input-dir if not absolute.")
    p.add_argument("--prior-file", default="input.dat",
                   help="Prior file (2 columns: upper bound, step), one row per patch. "
                        "Resolved against --input-dir if not absolute.")
    p.add_argument("--output-dir", default="./outputFiles",
                   help="Output directory (created if missing).")
    p.add_argument("--tag", default="p07_hz_fidu_fn1",
                   help="Tag inserted into all output filenames.")
    p.add_argument("--num-param", type=int, default=7,
                   help="Number of basis terms (P).")
    p.add_argument("--l100", type=int, default=10,
                   help="Grid points per dimension on each patch.")
    p.add_argument("--data-points", type=int, default=38,
                   help="Number of measurement rows to use "
                        "(default: all rows in --data-file).")
    p.add_argument("--total-rows", type=int, default=1339,
                   help="Number of patches in the prior file.")
    p.add_argument("--processes", type=int, default=0,
                   help="Worker processes (0 = all available cores).")
    p.add_argument("--chunk", type=int, default=200_000,
                   help="Grid combinations evaluated per chunk inside the worker.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the final overlay plot.")
    p.add_argument("--plot-show", action="store_true",
                   help="Also open an interactive window for the plot.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"--input-dir does not exist: {in_dir.resolve()}")

    # If the user passed an absolute path (or one that already includes a
    # directory component), respect it; otherwise resolve inside --input-dir.
    def _resolve(name: str) -> Path:
        path = Path(name)
        return path if path.is_absolute() else in_dir / path

    data_path = _resolve(args.data_file)
    prior_path = _resolve(args.prior_file)
    if not data_path.is_file():
        raise FileNotFoundError(f"data file not found: {data_path}")
    if not prior_path.is_file():
        raise FileNotFoundError(f"prior file not found: {prior_path}")

    P = args.num_param
    L = args.l100
    NR = args.total_rows
    tag = args.tag
    n_proc = args.processes if args.processes > 0 else cpu_count()

    # ---------------- Read inputs ---------------- #
    dat = np.loadtxt(data_path)
    if dat.ndim == 1 or dat.shape[1] != 3:
        raise ValueError(f"--data-file must have 3 columns, got shape {dat.shape}")

    # D defaults to the number of rows in the data file; if user provided a
    # value, cap-check it against the file length.
    if args.data_points is None:
        D = dat.shape[0]
    else:
        D = args.data_points
        if dat.shape[0] < D:
            raise ValueError(f"--data-file has {dat.shape[0]} rows, need at least {D}")
    dat = dat[:D]
    r_arr, hs_arr, err_arr = dat[:, 0], dat[:, 1], dat[:, 2]

    U = np.loadtxt(prior_path)
    if U.ndim == 1 or U.shape[1] != 2:
        raise ValueError(f"--prior-file must have 2 columns, got shape {U.shape}")
    if U.shape[0] < NR:
        raise ValueError(f"--prior-file has {U.shape[0]} rows, need at least {NR}")
    U = U[:NR]

    # First-pass basis: funct(r)^j for j = 0..P-1   -> shape (D, P)
    f_r = funct(r_arr)
    basis = np.vander(f_r, P, increasing=True)

    patch_args = [(float(U[i, 0]), float(U[i, 1])) for i in range(NR)]

    # ---------------- Stage 1: grid search in raw basis ---------------- #
    print(f"[stage 1] searching {L**P:,} combinations on {NR} patches "
          f"with {n_proc} processes ...", flush=True)

    dataDummy = np.zeros((P + 1, NR))
    init_args = (basis, hs_arr, err_arr, P, L, args.chunk)
    # with Pool(processes=n_proc, initializer=_init_worker, initargs=init_args) as pool:
    #     for i, (chi2, coeff) in enumerate(
    #             pool.imap(_patch_search, patch_args, chunksize=4)):
    #         dataDummy[0, i] = chi2
    #         dataDummy[1:, i] = coeff
    #         if (i + 1) % max(1, NR // 20) == 0:
    #             print(f"   patch {i + 1}/{NR}", flush=True)
    with Pool(processes=n_proc, initializer=_init_worker, initargs=init_args) as pool:
        for i, (chi2, coeff) in enumerate(tqdm(
                pool.imap(_patch_search, patch_args, chunksize=4),
                total=NR, desc="stage 1", unit="patch")):
            dataDummy[0, i] = chi2
            dataDummy[1:, i] = coeff


    np.savetxt(out_dir / f"DATA_{tag}.txt", dataDummy.T, fmt="%.10g")
    print("[stage 1] done.")

    # ---------------- Covariance matrix ---------------- #
    samples = dataDummy[1:, :]                                   # (P, NR)
    means = samples.mean(axis=1)
    diffs = samples - means[:, None]

    C = np.zeros((P, P))
    for j in range(P):
        C[j, j] = np.sum(diffs[j] ** 2) / (NR - 1)
    for i1 in range(1, P):
        for j in range(P - i1):
            rs = np.sum(samples[j] * samples[j + i1])
            C[j, j + i1] = (rs - means[j] * means[j + i1]) / (NR - 1)
            C[j + i1, j] = C[j, j + i1]

    np.savetxt(out_dir / f"COV_MATRIX_{tag}.txt", C, fmt="%.10g")

    # ---------------- Eigen-decomposition (DSYEV equivalent) ---------------- #
    eigvals, eigvecs = np.linalg.eigh(C)        # vectors are columns of eigvecs
    order = np.argsort(np.abs(eigvals))         # Fortran sorts by |lambda|, ascending
    DW = eigvals[order]
    DC = eigvecs[:, order]                      # columns = sorted eigenvectors

    with open(out_dir / f"EIGENVALUES_{tag}.txt", "w") as f:
        f.write(" ".join(f"{v:.10g}" for v in DW) + "\n")

    # Eigenvectors written one per row (i-th row = i-th eigenvector),
    # matching the original "stored column-wise" output.
    with open(out_dir / f"EIGENVECTORS_{tag}.txt", "w") as f:
        for i in range(P):
            f.write(" ".join(f"{v:.10g}" for v in DC[:, i]) + "\n")

    # ---------------- Eigenfunctions on the data grid ---------------- #
    # EF[i1, i] = sum_j DC[i1, j] * funct(r_i)^j
    EF = DC @ basis.T                                          # (P, D)

    with open(out_dir / f"EIGENFUNCTIONS_{tag}.txt", "w") as f:
        for i in range(D):
            f.write(" ".join(f"{v:.10g}" for v in EF[:, i]) + "\n")

    # Normalised: u_n[k, i] = EF[k, i] / sqrt( sum_m EF[m, i]^2 / err_i^2 )
    norm = np.sqrt(np.sum(EF ** 2 / err_arr ** 2, axis=0))
    U_N = EF / norm[None, :]
    with open(out_dir / f"NEIGENFNS_{tag}.txt", "w") as f:
        for i in range(D):
            f.write(" ".join(f"{v:.10g}" for v in U_N[:, i]) + "\n")

    # ---------------- Stage 2: grid search in eigenfunction basis ---------------- #
    basis2 = EF.T                                              # (D, P)

    print(f"[stage 2] searching {L**P:,} combinations on {NR} patches ...",
          flush=True)
    dataDummy2 = np.zeros((P + 1, NR))
    init_args2 = (basis2, hs_arr, err_arr, P, L, args.chunk)
    # with Pool(processes=n_proc, initializer=_init_worker, initargs=init_args2) as pool:
    #     for i, (chi2, coeff) in enumerate(
    #             pool.imap(_patch_search, patch_args, chunksize=4)):
    #         dataDummy2[0, i] = chi2
    #         dataDummy2[1:, i] = coeff
    #         if (i + 1) % max(1, NR // 20) == 0:
    #             print(f"   patch {i + 1}/{NR}", flush=True)
    with Pool(processes=n_proc, initializer=_init_worker, initargs=init_args2) as pool:
        for i, (chi2, coeff) in enumerate(tqdm(
                pool.imap(_patch_search, patch_args, chunksize=4),
                total=NR, desc="stage 2", unit="patch")):
            dataDummy2[0, i] = chi2
            dataDummy2[1:, i] = coeff


    np.savetxt(out_dir / f"DATA_FINAL_N_{tag}.txt", dataDummy2.T, fmt="%.10g")
    print("[stage 2] done.")

    # ---------------- Global minimum across patches ---------------- #
    j_best = int(np.argmin(dataDummy2[0]))
    dataStore = dataDummy2[:, j_best].copy()                         # (P+1,)
    with open(out_dir / f"FINAL_FILE_N_{tag}.txt", "w") as f:
        f.write(" ".join(f"{v:.10g}" for v in dataStore) + "\n")

    # ---------------- Reconstruction at several truncation levels ---------------- #
    # COFF(j) = sum_i RANBIR(i+1) * DC(i, j)   ->   coeffs_full = DC.T @ dataStore[1:]
    coeffs_full = DC.T @ dataStore[1:]                            # (P,)

    for k in range(P - 5, P + 1):
        if k < 1:
            continue
        coff_k = coeffs_full[:k]
        recon = basis[:, :k] @ coff_k                          # (D,)

        with open(out_dir / f"resultant_coff_{tag}_h0_{k}.txt", "w") as f:
            f.write(" ".join(f"{v:.10g}" for v in coff_k) + "\n")
        with open(out_dir / f"resultant_{tag}_h0_{k}.txt", "w") as f:
            for i in range(D):
                f.write(f"{r_arr[i]:.10g} {recon[i]:.10g}\n")

    # ---------------- Overlay plot ---------------- #
    if not args.no_plot:
        ks = [k for k in range(P - 5, P + 1) if k >= 1]
        plot_results(r_arr, hs_arr, err_arr, coeffs_full, ks, tag, out_dir,
                     show=args.plot_show)

    print(f"\nAll outputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
