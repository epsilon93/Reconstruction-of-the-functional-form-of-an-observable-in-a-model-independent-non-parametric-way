#!/usr/bin/env python3
"""
run_eigen.py -- single-entry-point launcher driven by params.in

Stage 0 (optional): generate_prior.py    -> input.dat                 (if generate_prior=true and Stage 1 is selected)
Stage 1           : pcaSourceCode.py     -> resultant_coff_<tag>_h0_<P>.txt etc.
Stage 2           : pymc_main.py         -> MCMC chain + corner plot

Which stages run is controlled by the `stages` key in params.in:

    stages = 1      -> only Stage 1
    stages = 2      -> only Stage 2 (Stage-1 outputs must already exist on disk)
    stages = 1,2    -> both (default)

Usage:
    python run_eigen.py                # reads ./params.in
    python run_eigen.py my_run.in      # reads an alternate file

None of the sub-scripts are modified by this wrapper; it just translates
`key = value` entries in the parameter file into the argparse flags each
script already understands.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------- #
#  Mapping: params.in key  ->  (CLI flag, kind)
#     kind = "value"  -> "<flag> <value>"  (skipped if value is blank)
#     kind = "flag"   -> "<flag>"          (only emitted when value is truthy)
# ---------------------------------------------------------------------- #
PCA_PARAM_MAP: dict[str, tuple[str, str]] = {
    "input_dir":    ("--input-dir",   "value"),
    "output_dir":   ("--output-dir",  "value"),
    "data_file":    ("--data-file",   "value"),
    "prior_file":   ("--prior-file",  "value"),
    "tag":          ("--tag",         "value"),
    "num_param":    ("--num-param",   "value"),
    "l100":         ("--l100",        "value"),
    "data_points":  ("--data-points", "value"),
    "total_rows":   ("--total-rows",  "value"),
    "processes":    ("--processes",   "value"),
    "chunk":        ("--chunk",       "value"),
    "no_plot":      ("--no-plot",     "flag"),
    "plot_show":    ("--plot-show",   "flag"),
}

PRIOR_PARAM_MAP: dict[str, tuple[str, str]] = {
    "prior_n_z":       ("--n-z",       "value"),
    "prior_n_dp":      ("--n-dp",      "value"),
    "prior_dp_min":    ("--dp-min",    "value"),
    "prior_dp_max":    ("--dp-max",    "value"),
    "prior_z_min":     ("--z-min",     "value"),
    "prior_z_max":     ("--z-max",     "value"),
    "prior_linear_dp": ("--linear-dp", "flag"),
    "l100":            ("--l100",      "value"),  # shared with PCA
}

PYMC_PARAM_MAP: dict[str, tuple[str, str]] = {
    "pymc_output_dir":       ("--output-dir",       "value"),
    "pymc_reduction":        ("--reduction",        "value"),
    "pymc_chi_num":          ("--chi-num",          "value"),
    "pymc_function_type":    ("--function-type",    "value"),
    "pymc_data_type":        ("--data-type",        "value"),
    "pymc_sim_points":       ("--sim-points",       "value"),
    "num_param":             ("--num-param",        "value"),     # shared

    # core sampler controls
    "pymc_NP":               ("--NP",               "value"),
    "pymc_tune":             ("--tune",             "value"),
    "pymc_chains":           ("--chains",           "value"),
    "pymc_cores":            ("--cores",            "value"),
    "pymc_seed":             ("--seed",             "value"),

    # sampler selection + per-sampler knobs
    "pymc_sampler":          ("--sampler",          "value"),
    "pymc_target_accept":    ("--target-accept",    "value"),
    "pymc_hmc_path_length":  ("--hmc-path-length",  "value"),
    "pymc_hmc_step_scale":   ("--hmc-step-scale",   "value"),
    "pymc_mh_scaling":       ("--mh-scaling",       "value"),
    "pymc_mh_tune_interval": ("--mh-tune-interval", "value"),

    # JAX-only knobs (used by nuts-numpyro / nuts-blackjax)
    "pymc_jax_chain_method": ("--jax-chain-method", "value"),

    # outputs
    "pymc_plot_file":        ("--plot-file",        "value"),
    "pymc_trace_file":       ("--trace-file",       "value"),
    "pymc_plot_vars":        ("--plot-vars",        "value"),
    "pymc_show":             ("--show",             "flag"),
}

# Keys consumed by the wrapper itself rather than passed through to a sub-script.
_WRAPPER_ONLY = {"generate_prior", "stages"}
ALL_KEYS = (set(PCA_PARAM_MAP) | set(PRIOR_PARAM_MAP)
            | set(PYMC_PARAM_MAP) | _WRAPPER_ONLY)


# ---------------------------------------------------------------------- #
#  Helpers
# ---------------------------------------------------------------------- #
_TRUE  = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0", ""}


def _to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def _parse_stages(value: str) -> set[int]:
    """Parse a `stages` spec into a set of stage numbers.

    Accepts:
      ""           -> {1, 2}    (default: run everything)
      "1"          -> {1}
      "2"          -> {2}
      "1,2" / "2,1" / "both" / "all" -> {1, 2}

    Raises ValueError on anything else.
    """
    v = value.strip().lower()
    if v in ("", "both", "all"):
        return {1, 2}

    parts: list[int] = []
    for token in v.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            n = int(token)
        except ValueError:
            raise ValueError(
                f"cannot interpret stages={value!r}. Use '1', '2', or '1,2'."
            )
        if n not in (1, 2):
            raise ValueError(
                f"unknown stage {n} in stages={value!r}; only 1 and 2 are valid."
            )
        parts.append(n)

    if not parts:
        raise ValueError(
            f"cannot interpret stages={value!r}. Use '1', '2', or '1,2'."
        )
    return set(parts)


def parse_params(path: Path) -> dict[str, str]:
    """Parse a params.in file into a {key: value_str} dict."""
    params: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(
                f"{path}:{lineno}: malformed line (no '='): {raw!r}"
            )
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"{path}:{lineno}: empty key in: {raw!r}")
        params[key] = value

    unknown = set(params) - ALL_KEYS
    if unknown:
        raise KeyError(
            f"unknown parameter key(s) in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(ALL_KEYS)}"
        )
    return params


def _build_cli(params: dict[str, str], mapping: dict[str, tuple[str, str]]
               ) -> list[str]:
    """Convert parsed params into CLI tokens using the given mapping."""
    argv: list[str] = []
    for key, value in params.items():
        if key not in mapping:
            continue
        flag, kind = mapping[key]
        if kind == "flag":
            if _to_bool(value):
                argv.append(flag)
        else:
            if value == "":          # blank -> let the sub-script use its own default
                continue
            argv.extend([flag, value])
    return argv


def _run(script: Path, cli: list[str], label: str) -> int:
    print(f"[run_eigen] {label}: {script.name} {' '.join(cli)}", flush=True)
    return subprocess.run([sys.executable, str(script), *cli]).returncode


# ---------------------------------------------------------------------- #
#  Main
# ---------------------------------------------------------------------- #
def main() -> int:
    params_path = Path(sys.argv[1] if len(sys.argv) > 1 else "params.in").expanduser()
    params_path = params_path.resolve()
    if not params_path.is_file():
        sys.exit(f"params file not found: {params_path}")

    here          = Path(__file__).resolve().parent
    pca_script    = here / "pcaSourceCode.py"
    prior_script  = here / "generate_prior.py"
    pymc_script   = here / "pymc_main.py"

    params = parse_params(params_path)
    print(f"[run_eigen] params file : {params_path}", flush=True)

    # --- Resolve which stages to run ---
    try:
        stages = _parse_stages(params.get("stages", ""))
    except ValueError as e:
        sys.exit(f"[run_eigen] {e}")
    print(f"[run_eigen] stages      : {sorted(stages)}\n", flush=True)

    # --- Stage 0: optional prior generation (only meaningful with Stage 1) ---
    want_prior = _to_bool(params.get("generate_prior", "false"))
    if want_prior and 1 not in stages:
        print("[run_eigen] note: generate_prior=true but Stage 1 is not "
              "selected; skipping Stage 0.\n", flush=True)
        want_prior = False

    if want_prior:
        if not prior_script.is_file():
            sys.exit(f"generate_prior.py not found (looked at {prior_script})")
        prior_cli = _build_cli(params, PRIOR_PARAM_MAP)
        input_dir  = params.get("input_dir", "")  or "."
        prior_file = params.get("prior_file", "") or "input.dat"
        prior_cli.extend(["--output", str(Path(input_dir) / prior_file)])
        rc = _run(prior_script, prior_cli, "stage 0 (prior generation)")
        if rc != 0:
            sys.exit(f"generate_prior.py failed with exit code {rc}")
        print(flush=True)

    # --- Stage 1: PCA / eigenfunction reconstruction ---
    if 1 in stages:
        if not pca_script.is_file():
            sys.exit(f"pcaSourceCode.py not found next to run_eigen.py "
                     f"(looked at {pca_script})")
        pca_cli = _build_cli(params, PCA_PARAM_MAP)
        rc = _run(pca_script, pca_cli, "stage 1 (PCA)")
        if rc != 0:
            sys.exit(f"pcaSourceCode.py failed with exit code {rc}")

    # --- Stage 2: PyMC MCMC + corner plot ---
    if 2 in stages:
        if not pymc_script.is_file():
            sys.exit(f"pymc_main.py not found (looked at {pymc_script})")

        # Construct the three input paths from Stage 1 outputs
        input_dir  = params.get("input_dir",  "") or "."
        output_dir = params.get("output_dir", "") or "."
        data_file  = params.get("data_file",  "") or "data_file.dat"
        tag        = params.get("tag",        "") or "run"
        num_param  = params.get("num_param",  "") or "7"

        data_path     = Path(input_dir)  / data_file
        data_ini_path = Path(output_dir) / f"DATA_{tag}.txt"
        coeff_path    = Path(output_dir) / f"resultant_coff_{tag}_h0_{num_param}.txt"

        # --- Pre-flight: if Stage 1 was skipped, its outputs must exist ---
        if 1 not in stages:
            missing = [p for p in (data_path, data_ini_path, coeff_path)
                       if not p.is_file()]
            if missing:
                sys.exit(
                    "Stage 2 was selected without Stage 1, but required input(s) "
                    "are missing on disk:\n"
                    + "\n".join(f"  - {m}" for m in missing)
                    + "\n\nFix one of:\n"
                      "  * run with stages=1 or stages=1,2 first to produce them\n"
                      "  * check that input_dir / output_dir / tag / num_param "
                      "in params.in match where the Stage-1 outputs actually live"
                )

        pymc_cli = _build_cli(params, PYMC_PARAM_MAP)
        pymc_cli.extend(["--data-file",     str(data_path)])
        pymc_cli.extend(["--data-ini-file", str(data_ini_path)])
        pymc_cli.extend(["--coeff-file",    str(coeff_path)])

        # Default pymc_output_dir to <output_dir>/pymc when blank
        if not params.get("pymc_output_dir", ""):
            pymc_cli.extend(["--output-dir", str(Path(output_dir) / "pymc")])

        print(flush=True)
        rc = _run(pymc_script, pymc_cli, "stage 2 (PyMC)")
        if rc != 0:
            sys.exit(f"pymc_main.py failed with exit code {rc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
