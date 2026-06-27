#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from grtp.viz.basin_map import plot_basin_map

_FATE_ORDER = ["escape_positive", "escape_negative", "plunge", "trapped", "failed"]

data_dir = Path("data")
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)

grids = {}
labels_to_load = [
    "a0.50_scaled_xpoint", "a0.90_scaled_xpoint", "a0.99_scaled_xpoint",
    "a0.90_E0.0_B0.0_xpoint", "a0.90_scaled_harris"
]

print("Loading saved HDF5 basin files...")
for label in labels_to_load:
    path = data_dir / f"exp2_basin_{label}.h5"
    if not path.exists():
        print(f"  [Warning] {path} not found. Will skip its plots.")
        continue
    with h5py.File(path, "r") as hf:
        fate_grid_int = hf["fate_grid_int"][:]
        r_grid = hf["r_grid"][:]
        ur_grid = hf["ur_grid"][:]
        M = hf.attrs["M"]
        a_val = hf.attrs["a"]
        r_X = hf.attrs["r_X"]
        E0_val = hf.attrs["E0"]
        f_type = hf.attrs["field_type"]
        
        fate_grid = np.empty_like(fate_grid_int, dtype="O")
        int_to_fate = {i: f for i, f in enumerate(_FATE_ORDER)}
        int_to_fate[4] = "failed"
        
        for i in range(fate_grid_int.shape[0]):
            for j in range(fate_grid_int.shape[1]):
                fate_grid[i, j] = int_to_fate.get(fate_grid_int[i, j], "failed")
        
        grids[label] = (fate_grid, r_grid, ur_grid, r_X, a_val, E0_val, f_type)
        print(f"  -> Loaded {label}")

print("\nGenerating multi-panel plots...")

# 1. Spin Sweep
if all(l in grids for l in ["a0.50_scaled_xpoint", "a0.90_scaled_xpoint", "a0.99_scaled_xpoint"]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#0D0D0D")
    labels = ["a0.50_scaled_xpoint", "a0.90_scaled_xpoint", "a0.99_scaled_xpoint"]
    for ax, lbl in zip(axes, labels):
        fg, rg, urg, rx, a_v, E_v, ft = grids[lbl]
        plot_basin_map(fate_grid=fg, r_values=rg, ur_values=urg, M=M, a=a_v, r_X=rx, ax=ax, title=rf"$a/M = {a_v:.2f}$ (Scaled X-point, $E_0={E_v:.2f}$)")
    fig.suptitle("Basin of Fate: Spin Sweep with Scaled E-Field", color="white", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_basin_spin_sweep.png", dpi=200)
    fig.savefig(fig_dir / "exp2_basin_spin_sweep.pdf")
    plt.close(fig)
    print("  -> Saved spin sweep figures.")

# 2. Reconnection Sweep
if all(l in grids for l in ["a0.90_E0.0_B0.0_xpoint", "a0.90_scaled_xpoint"]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0D0D0D")
    labels = ["a0.90_E0.0_B0.0_xpoint", "a0.90_scaled_xpoint"]
    titles = [r"Pure Geodesic ($E_0=0, B_0=0$)", r"Active Reconnection ($E_0=2.0$)"]
    for ax, lbl, title in zip(axes, labels, titles):
        fg, rg, urg, rx, a_v, E_v, ft = grids[lbl]
        plot_basin_map(fate_grid=fg, r_values=rg, ur_values=urg, M=M, a=a_v, r_X=rx, ax=ax, title=title)
    fig.suptitle(r"Basin of Fate: Reconnection E-Field Effect ($a/M=0.90$)", color="white", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_basin_reconnection_sweep.png", dpi=200)
    fig.savefig(fig_dir / "exp2_basin_reconnection_sweep.pdf")
    plt.close(fig)
    print("  -> Saved reconnection sweep figures.")

# 3. Geometry Sweep
if all(l in grids for l in ["a0.90_scaled_xpoint", "a0.90_scaled_harris"]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0D0D0D")
    labels = ["a0.90_scaled_xpoint", "a0.90_scaled_harris"]
    titles = [r"2D X-Point Field Geometry", r"1D Harris Neutral Sheet Geometry"]
    for ax, lbl, title in zip(axes, labels, titles):
        fg, rg, urg, rx, a_v, E_v, ft = grids[lbl]
        plot_basin_map(fate_grid=fg, r_values=rg, ur_values=urg, M=M, a=a_v, r_X=rx, ax=ax, title=title)
    fig.suptitle(r"Basin of Fate: Field Geometry Comparison ($a/M=0.90$, $E_0=2.0$)", color="white", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(fig_dir / "exp2_basin_geometry_sweep.png", dpi=200)
    fig.savefig(fig_dir / "exp2_basin_geometry_sweep.pdf")
    plt.close(fig)
    print("  -> Saved geometry sweep figures.")

print("\nRescue rendering complete!")
