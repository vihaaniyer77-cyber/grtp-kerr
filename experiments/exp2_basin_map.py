#!/usr/bin/env python3
"""
Experiment 2: Basin-of-fate map in the initial-condition phase space.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from grtp.spacetime.ergosphere import event_horizon, ergosphere_radius
from grtp.spacetime.kerr import metric as kerr_metric, zamo_tetrad
from grtp.fields.xpoint import XPointField, HarrisSheetField
from grtp.integrator.solver import ParticleIntegrator
from grtp.analysis.topology import FateClassifier, ESCAPE_POSITIVE, PLUNGE

def make_ic_equatorial(
    r0:   float,
    ur0:  float,
    M:    float,
    a:    float,
    uphi: float = 0.1,
) -> np.ndarray | None:
    """Generate state vector for equatorial motion."""
    theta = np.pi / 2.0
    g     = kerr_metric(r0, theta, M, a)

    A   = g[0, 0]
    gtp = g[0, 3]
    gpp = g[3, 3]
    grr = g[1, 1]

    if A > 0:
        coeff = gtp ** 2 - A * gpp
        if coeff > 0:
            uphi_min = np.sqrt(max(0.0, A * (grr * ur0 ** 2 + 1.0) / coeff))
            uphi = max(uphi, uphi_min * 1.05)
        else:
            return None

    B = 2.0 * gtp * uphi
    C = grr * ur0 * ur0 + gpp * uphi * uphi + 1.0
    if abs(A) < 1e-12:
        if abs(B) < 1e-14:
            return None
        ut = -C / B
    else:
        disc = B * B - 4.0 * A * C
        if disc < 0:
            return None
        ut = (-B - np.sqrt(disc)) / (2.0 * A)
    return np.array([0.0, r0, theta, 0.0, ut, ur0, 0.0, uphi])

def integrate_grid(
    a:           float,
    B0:          float,
    E0:          float,
    field_type:  str,  # "xpoint" or "harris"
    M:           float,
    lam:         float,
    N:           int,
    tau_max:     float,
    r_escape:    float,
    n_jobs:      int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r_plus = event_horizon(M, a)
    r_ergo = ergosphere_radius(np.pi / 2, M, a)
    r_X    = 0.5 * (r_plus + r_ergo)

    r_grid  = np.linspace(r_plus + 0.02, r_ergo + 0.8, N)
    ur_grid = np.linspace(-0.6, 0.6, N)

    if field_type == "xpoint":
        field = XPointField(M=M, a=a, r_X=r_X, B0=B0, E0=E0, lam=lam)
    elif field_type == "harris":
        field = HarrisSheetField(M=M, a=a, r_X=r_X, B0=B0, E0=E0, lam=lam, delta=0.2)
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    integ  = ParticleIntegrator(
        M=M, a=a, q_over_m=1.0, field=field,
        r_escape=r_escape, rtol=1e-9, atol=1e-11,
    )
    clf    = FateClassifier()

    y0_list = []
    valid   = []
    for r0 in r_grid:
        for ur0 in ur_grid:
            ic = make_ic_equatorial(r0, ur0, M, a)
            if ic is None:
                y0_list.append(None)
                valid.append(False)
            else:
                y0_list.append(ic)
                valid.append(True)

    valid_idx  = [i for i, v in enumerate(valid) if v]
    valid_y0   = [y0_list[i] for i in valid_idx]

    trajs_valid = integ.integrate_batch(valid_y0, tau_max=tau_max,
                                        n_out=100, n_jobs=n_jobs)

    all_fates = ["failed"] * (N * N)
    for list_idx, grid_idx in enumerate(valid_idx):
        traj = trajs_valid[list_idx]
        if traj is not None:
            all_fates[grid_idx] = clf.classify(traj)

    fate_grid = np.array(all_fates).reshape(N, N)
    return fate_grid, r_grid, ur_grid, r_X

def run(args: argparse.Namespace) -> None:
    """Run phase-space integration grid sweep and save plots."""
    M        = 1.0
    B0, E0   = args.B0, args.E0
    lam      = args.lam
    tau_max  = args.tau_max
    r_escape = args.r_escape
    n_jobs   = args.n_jobs
    N        = args.grid_size
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir  = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from grtp.viz.basin_map import plot_basin_map

    # Define the 5 configurations:
    # 1. Spin sweep (a=0.50, 0.90, 0.99) - X-Point, E0=2.0
    # 2. Reconnection sweep (E0=0.0 vs E0=2.0) - X-Point, a=0.90
    # 3. Geometry sweep (X-Point vs Harris Sheet) - a=0.90, E0=2.0

    print("==============================================================")
    print("Running Multi-Condition Basin of Fate Maps Sweep")
    print("==============================================================")

    grids = {}

    configs = [
        (0.90, 2.00, B0, "harris", "a0.90_scaled_harris"),
    ]

    for a_val, E0_val, B0_val, f_type, label in configs:
        print(f"\n---> Running configuration: a={a_val:.2f}, E0={E0_val:.1f}, B0={B0_val:.1f}, type={f_type}")
        fate_grid, r_grid, ur_grid, r_X = integrate_grid(
            a=a_val, B0=B0_val, E0=E0_val, field_type=f_type,
            M=M, lam=lam, N=N, tau_max=tau_max, r_escape=r_escape, n_jobs=n_jobs
        )
        grids[label] = (fate_grid, r_grid, ur_grid, r_X, a_val, E0_val, f_type)

        # Save individual configuration HDF5
        hdf_path = out_dir / f"exp2_basin_{label}.h5"
        with h5py.File(hdf_path, "w") as hf:
            hf.attrs["M"]       = M
            hf.attrs["a"]       = a_val
            hf.attrs["r_X"]     = r_X
            hf.attrs["B0"]      = B0
            hf.attrs["E0"]      = E0_val
            hf.attrs["lam"]     = lam
            hf.attrs["field_type"] = f_type
            hf.attrs["tau_max"] = tau_max
            hf.create_dataset("r_grid",    data=r_grid)
            hf.create_dataset("ur_grid",   data=ur_grid)
            from grtp.analysis.topology import _FATE_INT
            fate_int = np.vectorize(lambda f: _FATE_INT.get(f, 4))(fate_grid)
            hf.create_dataset("fate_grid_int", data=fate_int.astype(np.int8))
            fate_labels = "|".join(f"{k}:{v}" for k, v in _FATE_INT.items())
            hf.attrs["fate_encoding"] = fate_labels
        print(f"Saved HDF5 -> {hdf_path}")

    print("\n==============================================================")
    print("Harris Sheet configuration completed successfully!")
    print("==============================================================")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate basin of fate map.")
    p.add_argument("--spin",        type=float, default=0.9,    help="Black hole spin a/M")
    p.add_argument("--r-X",         type=float, default=None,   help="X-point radius (default: midpoint)")
    p.add_argument("--grid-size",   type=int,   default=300,    help="N×N grid size (N per axis)")
    p.add_argument("--tau-max",     type=float, default=300.0,  help="Max proper time [M]")
    p.add_argument("--B0",          type=float, default=4.0,    help="Magnetic field amplitude")
    p.add_argument("--E0",          type=float, default=2.0,    help="Reconnection E-field")
    p.add_argument("--lam",         type=float, default=0.4,    help="Current sheet half-width [M]")
    p.add_argument("--r-escape",    type=float, default=30.0,   help="Escape radius [M]")
    p.add_argument("--n-jobs",      type=int,   default=-1,     help="CPU workers (-1 = all)")
    p.add_argument("--out-dir",     type=str,   default="data",    help="HDF5 output directory")
    p.add_argument("--fig-dir",     type=str,   default="figures", help="Figure output directory")
    p.add_argument("--test",        action="store_true",
                   help="Quick test: 30×30 grid, tau_max=100")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.test:
        args.grid_size = 30
        args.tau_max   = 100.0
        print("[TEST MODE] 30×30 grid.")
    run(args)
