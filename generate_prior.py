#!/usr/bin/env python3
"""
generate_prior.py -- generate the prior-patch file `input.dat` consumed by
                     pcaSourceCode.py (Stage 1).

Each patch is a (r1_upper, step_size) pair such that the per-coefficient
candidate grid

    b_grid[i] = r1_upper - i * step_size,   i = 1..L100

straddles zero (so both positive and negative coefficient values are
candidates).  This is achieved by parametrising each patch with a
zero-position  z in (1, L100)  and a step size  dp,  and setting
r1_upper = z * dp.

Defaults (n_z=30, n_dp=45) produce 1350 patches with log-spaced step sizes
from 1e-3 to 1.0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_patches(l100, n_z, n_dp, dp_min, dp_max,
                     z_min=1.5, z_max=None, log_dp=True):
    if z_max is None:
        z_max = l100 - 0.5
    z_vals = np.linspace(z_min, z_max, n_z)
    if log_dp:
        dp_vals = np.geomspace(dp_min, dp_max, n_dp)
    else:
        dp_vals = np.linspace(dp_min, dp_max, n_dp)

    patches = []
    for z in z_vals:
        for dp in dp_vals:
            r1 = z * dp
            patches.append((r1, dp))
    return np.asarray(patches, dtype=np.float64)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--l100",     type=int,   default=10,
                   help="Per-coefficient grid resolution.")
    p.add_argument("--n-z",      type=int,   default=30,
                   help="Number of zero-position values.")
    p.add_argument("--n-dp",     type=int,   default=45,
                   help="Number of step-size values.")
    p.add_argument("--dp-min",   type=float, default=1e-3, help="Smallest step size.")
    p.add_argument("--dp-max",   type=float, default=1.0,  help="Largest step size.")
    p.add_argument("--z-min",    type=float, default=1.5,
                   help="Minimum zero-position (must be > 1).")
    p.add_argument("--z-max",    type=float, default=None,
                   help="Maximum zero-position (default: l100 - 0.5).")
    p.add_argument("--linear-dp", action="store_true",
                   help="Use linear step-size spacing instead of logarithmic.")
    p.add_argument("--output",   default="input.dat",
                   help="Output file path (parent dirs created automatically).")
    args = p.parse_args()

    z_max = args.z_max if args.z_max is not None else args.l100 - 0.5

    patches = generate_patches(
        args.l100, args.n_z, args.n_dp, args.dp_min, args.dp_max,
        z_min=args.z_min, z_max=z_max, log_dp=not args.linear_dp,
    )

    # Sanity check: every patch's b_grid straddles zero.
    arange = np.arange(1, args.l100 + 1, dtype=np.float64)
    bg = patches[:, 0:1] - arange[None, :] * patches[:, 1:2]
    has_pos = (bg.max(axis=1) > 0)
    has_neg = (bg.min(axis=1) < 0)
    assert np.all(has_pos & has_neg), "internal: not every row straddles zero"

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, patches, fmt="%.10g")

    print(f"[generate_prior] wrote {len(patches):,} patches -> {out_path}")
    print(f"[generate_prior] b_grid covers: "
          f"[{bg.min():.3g}, {bg.max():.3g}] across all patches "
          f"(each row straddles 0)")


if __name__ == "__main__":
    main()
