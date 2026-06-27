# GRTP-Kerr — General Relativistic Test-Particle Tracking in the Kerr Ergosphere

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Open-source framework for integrating charged test-particle trajectories in a
Kerr black hole spacetime subject to an analytical reconnection electromagnetic
field.  Developed in support of the paper:

> **"Orbital Topology and Energetics of Test Particles in Relativistic Magnetic
> Reconnection within the Kerr Ergosphere"**
> *Monthly Notices of the Royal Astronomical Society* (submitted)

---

## Physics Overview

The framework solves the **general-relativistic Lorentz force equation** in
Boyer-Lindquist (BL) coordinates:

```
du^μ/dτ + Γ^μ_αβ u^α u^β = (q/m) F^μ_ν u^ν
```

on a fixed Kerr background, superimposed with a localised analytical
**2D magnetic X-point** field mimicking a reconnection site inside the
ergosphere.  The conserved Killing energy

```
E_∞ = −(g_tt u^t + g_tφ u^φ) − (q/m) A_t
```

is tracked along every trajectory to quantify energy extraction via the
**magnetic Penrose process**.

---

## Installation (uv)

```bash
# Clone and install core dependencies
git clone https://github.com/vihaaniyer77-cyber/grtp-kerr.git
cd grtp-kerr
uv sync

# Optional: Numba JIT (~15-40x speedup on ODE hot loop)
uv sync --extra fast

# Optional: PyVista interactive 3D renders
uv sync --extra viz3d

# Development (sympy, pytest)
uv sync --extra dev
```

---

## Quick Start

```python
import numpy as np
from grtp.spacetime.ergosphere import event_horizon, ergosphere_radius
from grtp.fields.xpoint import XPointField
from grtp.integrator.solver import ParticleIntegrator

# --- Black hole parameters ---
M, a = 1.0, 0.9          # mass and spin (geometrized units)

# --- Place X-point just inside the ergosphere ---
r_ergo   = ergosphere_radius(np.pi / 2, M, a)
r_plus   = event_horizon(M, a)
r_X      = 0.5 * (r_plus + r_ergo)   # midpoint of ergosphere

field = XPointField(M=M, a=a, r_X=r_X, B0=1.0, E0=0.2, lam=0.5)

# --- Initial conditions: 8-vector (t, r, θ, φ, u^t, u^r, u^θ, u^φ) ---
# Start a proton at the X-point with a small inward radial kick
y0 = np.array([0.0, r_X, np.pi/2, 0.0,   # position
               2.0, -0.1, 0.0, 1.0])       # 4-velocity (u^t is corrected by normalisation; u^phi must satisfy co-rotation)

integrator = ParticleIntegrator(M=M, a=a, q_over_m=1.0, field=field,
                                r_escape=50.0)
traj = integrator.integrate(y0, tau_max=500.0)

print(f"Particle fate : {traj.fate}")
print(f"Initial E_∞   : {traj.E_inf[0]:.6f}")
print(f"Final   E_∞   : {traj.E_inf[-1]:.6f}")
print(f"ΔE/E_initial  : {(traj.E_inf[-1] - traj.E_inf[0]) / abs(traj.E_inf[0]):.4%}")
```

---

## Module Map

```
grtp/
├── spacetime/
│   ├── kerr.py          Metric g_μν, inverse g^μν, Christoffel Γ^μ_αβ,
│   │                    ZAMO tetrad e^μ_(a), frame-dragging ω
│   └── ergosphere.py    Horizon r_+, ergosphere r_E(θ), ISCO, boundary tests
│
├── fields/
│   ├── xpoint.py        XPointField (2D null-point, PRIMARY)
│   │                    HarrisSheetField (1D tanh profile, comparison)
│   └── potentials.py    4-potential A_μ via Coulomb-gauge line integration,
│                        fast pre-computed CubicSpline for hot loop use
│
├── integrator/
│   ├── equations.py     8-ODE RHS: dx^μ/dτ = u^μ,
│   │                               du^μ/dτ = −Γ u u + (q/m) F^μ_ν u^ν
│   └── solver.py        DOP853 wrapper, terminal events (horizon / escape),
│                        Trajectory dataclass, energy monitoring
│
├── analysis/            topology.py (fate classifier), energy.py
└── viz/                 basin_map.py, trajectory.py (visualisation pipeline)
```

---

## Experiments

| Script | Description |
|--------|-------------|
| `experiments/exp1_efficiency_sweep.py` | Energy extraction η vs. spin *a* and X-point radius r_X |
| `experiments/exp2_basin_map.py`        | 300×300 multi-condition basin-of-fate maps (escape / plunge / trapped) |

Run with `USE_NUMBA=1` for ~15–40× speedup on the ODE integrator hot loop.

---

## Validation

```bash
uv run pytest tests/ -v
```

Key tests:
- Schwarzschild ISCO circular orbit: |ΔE_∞| < 1e-8 over 100 orbits
- Kerr ISCO radius: matches Bardeen (1972) formula to < 0.01%
- Energy conservation (B≠0, E=0): |ΔE_∞/E_∞| < 1e-6 over full integration
- Flat-space straight-line geodesic (M=0, q=0): deviation < 1e-10

---

## Citation

If you use this code, please cite the accompanying MNRAS paper (BibTeX entry
will be added upon acceptance) and this repository via its Zenodo DOI.

---

## License

MIT © 2026 Your Name.  See [LICENSE](LICENSE).
