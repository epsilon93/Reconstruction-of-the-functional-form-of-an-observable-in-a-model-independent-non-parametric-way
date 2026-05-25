#!/usr/bin/env python3
"""
pymc_main.py -- Stage 2 of the eigenfunction reconstruction pipeline.

Reads the Stage-1 outputs (`DATA_<tag>.txt` for prior patches, and
`resultant_coff_<tag>_h0_<P>.txt` for the PCA coefficients) plus the
original H(z) data file, runs PyMC inference over the parameters

    OmegaM, H_C, Coff1, Coff2, Coff3, beta, sigma

and writes a corner plot (PNG) plus the trace (pickle).

Sampler choice
--------------
The MCMC method is selectable via --sampler:

    PyTensor backend (slow when pytensor.config.cxx="" on Apple Silicon):
        --sampler nuts            No-U-Turn Sampler
        --sampler hmc             Hamiltonian Monte Carlo
        --sampler metropolis      Metropolis-Hastings (gradient-free)

    JAX backend (fast on Apple Silicon; bypasses PyTensor C-backend issue):
        --sampler nuts-numpyro    NUTS via NumPyro + JAX
        --sampler nuts-blackjax   NUTS via BlackJAX + JAX

The JAX backends require additional packages:
    pip install numpyro jax jaxlib       # for nuts-numpyro
    pip install blackjax jax jaxlib      # for nuts-blackjax

Per-sampler knobs:
    --target-accept (NUTS, HMC, both JAX backends)
    --hmc-path-length, --hmc-step-scale (HMC only)
    --mh-scaling, --mh-tune-interval (Metropolis only)
    --jax-chain-method (JAX backends only; parallel | sequential | vectorized)

Notes
-----
- `pytensor.config.cxx = ""` is set BEFORE any pytensor / pymc import so
  that the macOS Apple-Silicon `-ld64` linker error doesn't fire.  This
  affects only the PyTensor-backed samplers; the JAX backends compile
  through XLA and are unaffected.
- `integrand(z, j)` and `Lambda(z)` depend only on the redshift array
  and the constants `One`, `Two` -- not on any sampled parameter -- so
  they're precomputed as numpy arrays before the model is built.
"""
from __future__ import annotations

# ---- PyTensor C-backend kill switch (macOS Apple Silicon) --------------
# Must come BEFORE any pytensor / pymc import.
import pytensor
pytensor.config.cxx = ""
# -----------------------------------------------------------------------

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.integrate
from scipy.interpolate import UnivariateSpline as intS

import pytensor.tensor as tt
import pymc as pm

import arviz as az
import arviz.labels as azl

import matplotlib
matplotlib.use("Agg")           # headless-safe; --show overrides below
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------- #
#  Constants and labels
# ---------------------------------------------------------------------- #
PARAM_LABELS_TEX = {
    "OmegaM": r"$\Omega_m$",
    "Coff1":  r"$\alpha_0$",
    "H_C":    r"$h_0$",
    "Coff2":  r"$\alpha_1$",
    "Coff3":  r"$\alpha_2$",
    "beta":   r"$\beta$",
    "sigma":  r"$\sigma$",
}

# Physical-matter-density constraint: Omega_m * h^2 = OM_H2_PRIOR.
# This fixes h_0 once Omega_m is sampled, so the model treats H_C as a
# deterministic function of OmegaM rather than an independent parameter.
OM_H2_PRIOR = 0.14314

# Sampler name groups (canonical names; argparse `choices` enforces these).
_PYTENSOR_SAMPLERS = {"nuts", "hmc", "metropolis"}
_JAX_SAMPLERS      = {"nuts-numpyro", "nuts-blackjax"}
_ALL_SAMPLERS      = sorted(_PYTENSOR_SAMPLERS | _JAX_SAMPLERS)


# ---------------------------------------------------------------------- #
#  CLI
# ---------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- input files ----
    p.add_argument("--data-file", required=True,
                   help="Hubble data: 3 cols (z, H, sigma_H).")
    p.add_argument("--data-ini-file", required=True,
                   help="Stage-1 prior patches DATA_<tag>.txt (chi^2, c1..cP).")
    p.add_argument("--coeff-file", required=True,
                   help="Stage-1 PCA coefficients resultant_coff_<tag>_h0_<P>.txt.")

    # ---- output ----
    p.add_argument("--output-dir", default="./outputFiles/pymc",
                   help="Directory for plot + trace pickle (created if missing).")
    p.add_argument("--plot-file",  default="pymc_pairplot.png")
    p.add_argument("--trace-file", default="pymc_trace.pkl")
    p.add_argument("--plot-vars",  default="OmegaM,Coff1,Coff2,Coff3,beta",
                   help="Comma-separated variables to corner-plot. "
                        "H_C is a deterministic function of OmegaM, so "
                        "it's omitted by default to avoid a redundant "
                        "perfect-curve panel.")
    p.add_argument("--show", action="store_true",
                   help="Open an interactive plot window (in addition to PNG).")

    # ---- model ----
    p.add_argument("--num-param",     type=int,   default=7)
    p.add_argument("--reduction",     type=int,   default=1,
                   help="n_p_f = num_param - reduction.")
    p.add_argument("--chi-num",       type=int,   default=33,
                   help="chi^2 cut on prior patches.")
    p.add_argument("--function-type", type=int,   default=1, choices=[1, 2],
                   help="1 -> (One,Two)=(1,1); 2 -> (0,1).")
    p.add_argument("--data-type",     type=int,   default=0, choices=[0, 1],
                   help="0 = real data; 1 = simulated/fiducial.")
    p.add_argument("--sim-points",    type=int,   default=100,
                   help="Number of synthetic redshift points.")

    # ---- sampler shared knobs ----
    p.add_argument("--NP",       type=int, default=2000,
                   help="Posterior draws per chain.")
    p.add_argument("--tune",     type=int, default=1000,
                   help="Tuning steps per chain.")
    p.add_argument("--chains",   type=int, default=2)
    p.add_argument("--cores",    type=int, default=1,
                   help="(PyTensor backends only; JAX uses --jax-chain-method.)")
    p.add_argument("--seed",     type=int, default=8927)

    # ---- sampler selection ----
    p.add_argument("--sampler", type=str, default="nuts",
                   choices=_ALL_SAMPLERS,
                   help=("MCMC method: PyTensor-backend ('nuts', 'hmc', "
                         "'metropolis') or JAX-backend "
                         "('nuts-numpyro', 'nuts-blackjax'). "
                         "JAX backends are typically much faster on Apple "
                         "Silicon when the PyTensor C backend is disabled."))

    p.add_argument("--target-accept", type=float, default=0.9,
                   help="NUTS/HMC target acceptance rate (default 0.9). "
                        "Ignored for metropolis.")

    # HMC-only
    p.add_argument("--hmc-path-length", type=float, default=None,
                   help="HMC trajectory length (default: PyMC 2.0).")
    p.add_argument("--hmc-step-scale",  type=float, default=None,
                   help="HMC initial step size scale (default: PyMC 0.25).")

    # Metropolis-only
    p.add_argument("--mh-scaling",       type=float, default=None,
                   help="Metropolis proposal scaling (default: PyMC 1.0).")
    p.add_argument("--mh-tune-interval", type=int,   default=None,
                   help="Metropolis adaptation interval (default: PyMC 100).")

    # JAX-only
    p.add_argument("--jax-chain-method", type=str, default="parallel",
                   choices=["parallel", "sequential", "vectorized"],
                   help="How JAX runs multiple chains. 'parallel' (default) "
                        "runs each chain on a separate device; 'vectorized' "
                        "vmap's them on one device; 'sequential' runs one at "
                        "a time. JAX backends only; ignored otherwise.")

    return p.parse_args()


# ---------------------------------------------------------------------- #
#  Sampler dispatch
# ---------------------------------------------------------------------- #
def build_step_method(sampler_name: str, args: argparse.Namespace):
    """Construct a PyTensor-backed step method.

    Used only for --sampler in {nuts, hmc, metropolis}.  Must be called
    inside a `with model:` block.

    Returns
    -------
    step : pm.NUTS | pm.HamiltonianMC | pm.Metropolis
    """
    name = sampler_name.lower().strip()

    if name in ("nuts", "no-u-turn", "no_u_turn"):
        return pm.NUTS(target_accept=args.target_accept)

    if name in ("hmc", "hamiltonianmc", "hamiltonian", "hamiltonian-mc"):
        kwargs = {"target_accept": args.target_accept}
        if args.hmc_path_length is not None:
            kwargs["path_length"] = args.hmc_path_length
        if args.hmc_step_scale is not None:
            kwargs["step_scale"] = args.hmc_step_scale
        return pm.HamiltonianMC(**kwargs)

    if name in ("metropolis", "mh", "metropolis-hastings"):
        kwargs: dict = {}
        if args.mh_scaling is not None:
            kwargs["scaling"] = args.mh_scaling
        if args.mh_tune_interval is not None:
            kwargs["tune_interval"] = args.mh_tune_interval
        return pm.Metropolis(**kwargs)

    raise ValueError(
        f"build_step_method got unsupported sampler {sampler_name!r}; "
        "JAX-backed samplers go through run_sampling()."
    )


def run_sampling(model: pm.Model, args: argparse.Namespace):
    """Dispatch to the correct sampler and return its trace.

    Returns
    -------
    trace : pm.backends.MultiTrace  (PyTensor backends)
            or arviz.InferenceData  (JAX backends)

    Downstream code uses _get_flat_var() / _var_names() to handle both.
    """
    name = args.sampler.lower().strip()

    # ---- JAX-backed NUTS (NumPyro) ----
    if name in ("nuts-numpyro", "nuts_numpyro", "numpyro"):
        try:
            from pymc.sampling.jax import sample_numpyro_nuts
        except ImportError as e:
            raise ImportError(
                "--sampler nuts-numpyro requires numpyro and jax.\n"
                "Install with: pip install numpyro jax jaxlib"
            ) from e
        with model:
            print(f"[pymc_main] Sampler: NUTS-NUMPYRO  "
                  f"(draws={args.NP}, tune={args.tune}, "
                  f"chains={args.chains}, "
                  f"chain_method={args.jax_chain_method!r})")
            return sample_numpyro_nuts(
                draws=args.NP,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
                random_seed=args.seed,
                chain_method=args.jax_chain_method,
                progressbar=True,
            )

    # ---- JAX-backed NUTS (BlackJAX) ----
    if name in ("nuts-blackjax", "nuts_blackjax", "blackjax"):
        try:
            from pymc.sampling.jax import sample_blackjax_nuts
        except ImportError as e:
            raise ImportError(
                "--sampler nuts-blackjax requires blackjax and jax.\n"
                "Install with: pip install blackjax jax jaxlib"
            ) from e
        with model:
            print(f"[pymc_main] Sampler: NUTS-BLACKJAX  "
                  f"(draws={args.NP}, tune={args.tune}, "
                  f"chains={args.chains}, "
                  f"chain_method={args.jax_chain_method!r})")
            return sample_blackjax_nuts(
                draws=args.NP,
                tune=args.tune,
                chains=args.chains,
                target_accept=args.target_accept,
                random_seed=args.seed,
                chain_method=args.jax_chain_method,
                progressbar=True,
            )

    # ---- PyTensor-backed step methods ----
    with model:
        step = build_step_method(name, args)
        print(f"[pymc_main] Sampler: {name.upper()}  "
              f"(draws={args.NP}, tune={args.tune}, chains={args.chains})")
        # When step= is explicit, target_accept lives on the step instance;
        # do NOT also pass it to pm.sample.
        #
        # return_inferencedata=True so the result is an arviz.InferenceData,
        # matching what the JAX backends return.  Recent arviz versions
        # (>=0.20) dropped MultiTrace from the accepted input types of
        # az.summary / az.hdi / az.plot_pair, so MultiTrace would now raise
        # `ValueError: Can only convert ... to InferenceData, not MultiTrace`.
        return pm.sample(
            draws=args.NP,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            step=step,
            random_seed=args.seed,
            return_inferencedata=True,
        )


# ---------------------------------------------------------------------- #
#  Trace adapters: handle MultiTrace and InferenceData uniformly
# ---------------------------------------------------------------------- #
def _is_idata(trace) -> bool:
    """True if `trace` is an arviz.InferenceData.

    With return_inferencedata=True for pm.sample and the JAX backends always
    returning InferenceData, this is now true for every backend.  The branch
    is kept for defensive compatibility with any future code path that might
    yield a MultiTrace.
    """
    return hasattr(trace, "posterior")


def _var_names(trace) -> list[str]:
    if _is_idata(trace):
        return list(trace.posterior.data_vars)
    return list(trace.varnames)


def _get_flat_var(trace, name: str):
    """Return a flat 1-D numpy array of samples for `name`, or None if absent."""
    if _is_idata(trace):
        if name not in trace.posterior:
            return None
        # InferenceData posterior has shape (chain, draw, [...])
        return np.asarray(trace.posterior[name].values).reshape(-1)
    if name not in trace.varnames:
        return None
    return np.asarray(trace[name])


# ---------------------------------------------------------------------- #
#  Main
# ---------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- function-type controls (One, Two) used inside fun1 ----
    if args.function_type == 1:
        One, Two = 1, 1
        one_sym, two_sym = 1, 1
    else:  # function_type == 2
        One, Two = 0, 1
        one_sym, two_sym = 0, 1

    # ---- Load Stage-1 outputs ----
    data_ini = np.loadtxt(args.data_ini_file)         # (NR, 1+P) : chi^2, c1..cP
    dat      = np.loadtxt(args.data_file)             # (Nz, 3)   : z, H, sigma_H
    coeff    = np.loadtxt(args.coeff_file)            # (P,)      : PCA coefficients

    num_param = args.num_param
    reduction = args.reduction
    n_p_f     = num_param - reduction

    # ---- Filter prior patches by chi^2 cut ----
    nr = data_ini.shape[0]
    keep_mask = data_ini[:, 0] <= args.chi_num
    new_dat_pca = data_ini[keep_mask, 1:]
    print(f"[pymc_main] kept {keep_mask.sum()} / {nr} prior patches "
          f"with chi^2 <= {args.chi_num}")

    # ---- Mean and covariance of kept coefficients ----
    nr_eff = new_dat_pca.shape[0]
    mean = new_dat_pca.mean(axis=0)
    cov  = np.empty((num_param, num_param))
    for j in range(num_param):
        cov[j, j] = ((new_dat_pca[:, j] - mean[j]) ** 2).sum() / (nr_eff - 1)
    for i1 in range(num_param - 1):
        for j in range(num_param - i1):
            rs = (new_dat_pca[:, j] * new_dat_pca[:, j + i1]).sum()
            cov[j + i1, j] = (rs - mean[j] * mean[j + i1]) / (nr_eff - 1)
            cov[j, j + i1] = cov[j + i1, j]

    # ---- Eigendecomposition (replaces Fortran DSYEV) ----
    e_val, e_vec = np.linalg.eigh(cov)
    # Build the (P+1, P) array [[lambda_1, lambda_2, ...],
    #                            [v_1     , v_2     , ...]] expected downstream
    e_vae = np.vstack([e_val.reshape(1, -1), e_vec])

    # ---- ERR function for H(z) (truncated to n_p_f basis terms) ----
    from sympy import lambdify, sqrt as sym_sqrt
    from sympy.abc import x as x_sym

    def fnct(x1, a, b):
        return (x1 ** a) / ((1 + x1) ** b)

    err = 0
    for i1 in range(n_p_f):
        e_fn = 0
        for j in range(n_p_f):
            e_fn = e_fn + e_vae[j + 1, i1] * (fnct(x_sym, one_sym, two_sym) ** j)
        err = (e_fn ** 2) * e_vae[0, i1]
    err = sym_sqrt(err)
    err_fn = lambdify(x_sym, err)

    # ---- Hubble reconstruction at given truncation level ----
    def Hubble(rs, redu):
        ii1 = num_param - redu
        s = 0.0
        for j_f in range(ii1):
            s = s + coeff[j_f] * (fnct(rs, one_sym, two_sym) ** j_f)
        return s

    # ---- Build sim_data ----
    err_spl = intS(dat[:, 0], dat[:, 2])
    sim_points_num = args.sim_points
    sim_data = np.empty((sim_points_num, 5), dtype=float)
    pc_red = 0
    sim_data[:, 0] = np.linspace(dat[:, 0].min(), dat[:, 0].max(), sim_points_num)
    sim_data[:, 1] = Hubble(sim_data[:, 0], pc_red)
    sim_data[:, 2] = Hubble(sim_data[:, 0], pc_red + 1)
    sim_data[:, 3] = Hubble(sim_data[:, 0], pc_red + 2)
    sim_data[:, 4] = err_spl(sim_data[:, 0])

    # ---- Integrand and Lambda (precomputed; no sampled params inside) ----
    def fun1(redf, order):
        epsilon = np.log(1 + redf)
        return (((np.exp(epsilon) - 1) ** One) / (np.exp(epsilon)) ** Two)

    def integrand(up_redshift, order):
        up_eps = np.log(1 + up_redshift)
        val = scipy.integrate.quad(fun1, 0, up_eps, args=(order,))[0]
        return val ** order

    def LambdaD(redshift):
        return 1.0 / (1.0 + redshift)

    def Lambda(redshift):
        return scipy.integrate.quad(LambdaD, 0, redshift)[0]

    redshift = sim_data[:, 0].astype(np.float64)
    integrand_1 = np.array([integrand(z, 1) for z in redshift], dtype=np.float64)
    integrand_2 = np.array([integrand(z, 2) for z in redshift], dtype=np.float64)
    integrand_3 = np.array([integrand(z, 3) for z in redshift], dtype=np.float64)
    Lambda_z    = np.array([Lambda(z)      for z in redshift], dtype=np.float64)

    # ---- PyMC model ----
    basic_model = pm.Model()

    with basic_model:
        OmegaM = pm.Uniform("OmegaM", lower=0.1, upper=0.6)
        Coff1  = pm.Normal ("Coff1",  mu=0.3,  sigma=1.0)
        Coff2  = pm.Normal ("Coff2",  mu=0.0,  sigma=2.5)
        Coff3  = pm.Normal ("Coff3",  mu=0.0,  sigma=3.0)
        beta   = pm.Uniform("beta",   lower=0.0, upper=1.0)
        sigma  = pm.HalfNormal("sigma", sigma=1.0)

        # H_C is no longer sampled independently; it is fixed by the
        # physical-matter-density constraint OmegaM * h^2 = OM_H2_PRIOR,
        # i.e. h = sqrt(OM_H2_PRIOR / OmegaM).  Wrapped in pm.Deterministic
        # so it is still tracked in the trace and shows up in the corner
        # plot, but contributes zero sampled dimensions.
        H_C = pm.Deterministic("H_C", tt.sqrt(OM_H2_PRIOR / OmegaM))

        dmy2 = Coff1 * integrand_1 + Coff2 * integrand_2 + Coff3 * integrand_3
        dmy3 = tt.exp(3.0 * (dmy2 + Lambda_z))
        inte = H_C * tt.sqrt(OmegaM * (1.0 + redshift) ** 3
                             + (1.0 - OmegaM) * dmy3)
        predict = inte * beta

        likes = pm.Normal("likes", mu=predict, sigma=sigma,
                          observed=sim_data[:, 1])

    # ---- Run sampler (dispatches across PyTensor / JAX backends) ----
    trace = run_sampling(basic_model, args)

    # ---- Diagnostics (az.summary / az.hdi accept both MultiTrace and IData) ----
    print("=" * 60)
    print(az.summary(trace, round_to=2, kind="all"))
    print("--- 1 sigma ---")
    print(az.hdi(trace, hdi_prob=0.68))
    print("--- 2 sigma ---")
    print(az.hdi(trace, hdi_prob=0.95))
    print("=" * 60)

    # ---- Corner plot ----
    plot_path = out_dir / args.plot_file
    plt.rcParams["figure.constrained_layout.use"] = True

    plot_vars = [v.strip() for v in args.plot_vars.split(",") if v.strip()]
    labeller  = azl.MapLabeller(var_name_map={k: PARAM_LABELS_TEX.get(k, k)
                                              for k in plot_vars})

    az.plot_pair(
        trace,
        var_names=plot_vars,
        group="posterior",     # required for InferenceData
        labeller=labeller,
        kind="kde",
        marginals=True,
        textsize=22,
        point_estimate="mode",
    )
    plt.savefig(plot_path, dpi=150)
    print(f"[pymc_main] wrote {plot_path}")
    if args.show:
        plt.show()
    plt.close("all")

    # ---- Save trace ----
    trace_path = out_dir / args.trace_file
    trace_dict = {}
    for k in PARAM_LABELS_TEX:
        arr = _get_flat_var(trace, k)
        if arr is not None:
            trace_dict[k] = arr

    with open(trace_path, "wb") as fh:
        pickle.dump({
            "param_names": list(PARAM_LABELS_TEX),
            "trace_dict":  trace_dict,
            "sampler":     args.sampler,
            "draws":       args.NP,
            "tune":        args.tune,
            "chains":      args.chains,
            "seed":        args.seed,
            "backend":     "jax" if _is_idata(trace) else "pytensor",
        }, fh)
    print(f"[pymc_main] wrote {trace_path}")

    elapsed = time.perf_counter() - start
    print(f"[pymc_main] total time: {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
