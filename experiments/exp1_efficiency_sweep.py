#!/usr/bin/env python3
"""
Experiment 1: Sweep spin (a/M) and X-point radius (r_X) to evaluate energy extraction efficiency.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from grtp.spacetime.ergosphere import event_horizon, ergosphere_radius
from grtp.spacetime.kerr import zamo_tetrad, metric
from grtp.fields.xpoint import XPointField
from grtp.integrator.solver import ParticleIntegrator
from grtp.analysis.topology import FateStats, FateClassifier

def sample_thermal_ic(
    r_X:       float,
    theta_X:   float,
    M:         float,
    a:         float,
    kT:        float = 0.05,
    rng:       np.random.Generator = None,
) -> np.ndarray:
    """Sample ZAMO-frame thermal velocity and return state vector."""
    if rng is None:
        rng = np.random.default_rng()

    v_r_hat   = rng.normal(0.0, np.sqrt(kT))
    v_th_hat  = rng.normal(0.0, np.sqrt(kT))
    v_ph_hat  = rng.normal(0.0, np.sqrt(kT))
    v2 = v_r_hat**2 + v_th_hat**2 + v_ph_hat**2

    if v2 >= 0.85**2:
        v_scale = 0.80 / np.sqrt(v2)
        v_r_hat  *= v_scale
        v_th_hat *= v_scale
        v_ph_hat *= v_scale
        v2        = v_r_hat**2 + v_th_hat**2 + v_ph_hat**2

    gamma = 1.0 / np.sqrt(1.0 - v2)
    tet   = zamo_tetrad(r_X, theta_X, M, a)

    u = (gamma * tet[0]
         + gamma * v_r_hat  * tet[1]
         + gamma * v_th_hat * tet[2]
         + gamma * v_ph_hat * tet[3])

    return np.array([0.0, r_X, theta_X, 0.0, u[0], u[1], u[2], u[3]])

def run_grid_point(
    a:           float,
    r_X:         float,
    M:           float,
    B0:          float,
    E0:          float,
    lam:         float,
    n_particles: int,
    tau_max:     float,
    kT:          float,
    r_escape:    float,
    n_jobs:      int,
    seed:        int,
) -> FateStats:
    """Evaluate a single grid point in parallel."""
    rng   = np.random.default_rng(seed)
    field = XPointField(M=M, a=a, r_X=r_X, B0=B0, E0=E0, lam=lam)
    integ = ParticleIntegrator(
        M=M, a=a, q_over_m=1.0, field=field,
        r_escape=r_escape, rtol=1e-9, atol=1e-11,
    )
    theta_X = np.pi / 2.0
    y0_list = [
        sample_thermal_ic(r_X, theta_X, M, a, kT=kT, rng=rng)
        for _ in range(n_particles)
    ]
    trajs = integ.integrate_batch(y0_list, tau_max=tau_max,
                                  n_out=200, n_jobs=n_jobs)
    return FateStats.from_trajectories(trajs, FateClassifier())

def run(args: argparse.Namespace) -> None:
    """Execute grid sweep, save HDF5, and generate heatmap."""
    M        = 1.0
    B0, E0   = args.B0, args.E0
    lam      = args.lam
    tau_max  = args.tau_max
    kT       = args.kT
    r_escape = args.r_escape
    n_jobs   = args.n_jobs
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir  = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    a_values  = np.linspace(0.1, 0.99, args.n_spin)
    f_values  = np.linspace(0.1, 0.9, args.n_rx)

    print(f"\n{'='*60}")
    print("Experiment 1: Energy Extraction Efficiency Sweep")
    print(f"  Spin grid   : {args.n_spin} values in [0.1, 0.99]")
    print(f"  r_X grid    : {args.n_rx} fractional positions in ergosphere")
    print(f"  N_particles : {args.n_particles} per grid point")
    print(f"  tau_max     : {tau_max} M")
    print(f"  Workers     : {n_jobs}")
    print(f"  Total runs  : {args.n_spin * args.n_rx * args.n_particles}")
    print(f"{'='*60}\n")

    efficiency_grid     = np.zeros((args.n_spin, args.n_rx))
    mean_gain_grid      = np.zeros((args.n_spin, args.n_rx))
    plunge_fraction_grid = np.zeros((args.n_spin, args.n_rx))
    rX_grid             = np.zeros((args.n_spin, args.n_rx))

    t0 = time.time()

    for i, a in enumerate(tqdm(a_values, desc="Spin loop")):
        r_plus = event_horizon(M, a)
        r_ergo = ergosphere_radius(np.pi / 2, M, a)
        r_X_arr = r_plus + f_values * (r_ergo - r_plus)

        for j, r_X in enumerate(tqdm(r_X_arr, desc=f"  a={a:.2f} r_X loop",
                                      leave=False)):
            rX_grid[i, j] = r_X
            try:
                stats = run_grid_point(
                    a=a, r_X=r_X, M=M,
                    B0=B0, E0=E0, lam=lam,
                    n_particles=args.n_particles,
                    tau_max=tau_max, kT=kT,
                    r_escape=r_escape, n_jobs=n_jobs,
                    seed=42 + i * 100 + j,
                )
                efficiency_grid[i, j]      = stats.efficiency
                mean_gain_grid[i, j]       = stats.mean_energy_gain
                plunge_fraction_grid[i, j] = stats.n_plunge / max(stats.n_total, 1)
            except Exception as exc:
                print(f"\nWARN: (a={a:.2f}, r_X={r_X:.3f}) failed: {exc}")
                efficiency_grid[i, j] = float("nan")

    elapsed = time.time() - t0
    print(f"\nTotal compute time: {elapsed/60:.1f} min")

    hdf_path = out_dir / "exp1_efficiency.h5"
    with h5py.File(hdf_path, "w") as hf:
        hf.attrs["M"]           = M
        hf.attrs["B0"]          = B0
        hf.attrs["E0"]          = E0
        hf.attrs["lam"]         = lam
        hf.attrs["tau_max"]     = tau_max
        hf.attrs["n_particles"] = args.n_particles
        hf.attrs["kT"]          = kT
        hf.create_dataset("a_values",            data=a_values)
        hf.create_dataset("f_values",            data=f_values)
        hf.create_dataset("rX_grid",             data=rX_grid)
        hf.create_dataset("efficiency",          data=efficiency_grid)
        hf.create_dataset("mean_energy_gain",    data=mean_gain_grid)
        hf.create_dataset("plunge_fraction",     data=plunge_fraction_grid)
    print(f"Saved HDF5 → {hdf_path}")

    import matplotlib.pyplot as plt
    from grtp.viz.basin_map import plot_efficiency_heatmap

    # Normalization correction: isolate the frame-dragging contribution
    # by subtracting the purely EM-accelerated baseline at the lowest spin (index 0)
    baseline_efficiency = efficiency_grid[0, :].copy()
    norm_efficiency = efficiency_grid - baseline_efficiency[np.newaxis, :]

    vmax_dynamic = max(0.05, float(np.nanmax(norm_efficiency)))
    fig     = plot_efficiency_heatmap(
        efficiency_grid  = norm_efficiency,
        a_values         = a_values,
        rX_values        = rX_grid,
        vmax             = vmax_dynamic,
        title            = (r"Normalized Penrose Efficiency $\eta_{norm}$  vs  Spin $a/M$  and  "
                            r"X-point Radius $r_X$"),
    )
    fig_path = fig_dir / "exp1_efficiency.pdf"
    fig.savefig(fig_path)
    fig.savefig(fig_dir / "exp1_efficiency.png", dpi=200)
    print(f"Saved figure → {fig_path}")
    plt.close(fig)

    # Plot 2: Mean Energy Gain
    vmax_gain = max(0.05, float(np.nanmax(mean_gain_grid)))
    fig_gain = plot_efficiency_heatmap(
        efficiency_grid  = mean_gain_grid,
        a_values         = a_values,
        rX_values        = rX_grid,
        vmax             = vmax_gain,
        title            = (r"Mean Energy Gain $\Delta E / E_0$  vs  Spin $a/M$  and  "
                            r"X-point Radius $r_X$"),
        cbar_label       = r"Mean Energy Gain $\Delta E / E_0$",
    )
    fig_gain_path = fig_dir / "exp1_energy_gain.pdf"
    fig_gain.savefig(fig_gain_path)
    fig_gain.savefig(fig_dir / "exp1_energy_gain.png", dpi=200)
    print(f"Saved figure → {fig_gain_path}")
    plt.close(fig_gain)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid sweep for energy extraction efficiency.")
    p.add_argument("--n-spin",       type=int,   default=30,    help="Number of spin values")
    p.add_argument("--n-rx",         type=int,   default=15,     help="Number of r_X positions")
    p.add_argument("--n-particles",  type=int,   default=150,   help="Particles per grid point")
    p.add_argument("--tau-max",      type=float, default=300.0, help="Max proper time [M]")
    p.add_argument("--kT",           type=float, default=0.05,  help="Thermal energy scale")
    p.add_argument("--B0",           type=float, default=1.0,   help="Magnetic field amplitude")
    p.add_argument("--E0",           type=float, default=0.2,   help="Reconnection E-field")
    p.add_argument("--lam",          type=float, default=0.4,   help="Current sheet half-width [M]")
    p.add_argument("--r-escape",     type=float, default=30.0,  help="Escape radius [M]")
    p.add_argument("--n-jobs",       type=int,   default=-1,    help="CPU workers (-1 = all)")
    p.add_argument("--out-dir",      type=str,   default="data",    help="HDF5 output directory")
    p.add_argument("--fig-dir",      type=str,   default="figures", help="Figure output directory")
    p.add_argument("--test",         action="store_true",
                   help="Quick test: 3 spins × 3 r_X × 20 particles")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.test:
        args.n_spin       = 3
        args.n_rx         = 3
        args.n_particles  = 20
        args.tau_max      = 100.0
        print("[TEST MODE] Running with reduced grid.")
    run(args)
