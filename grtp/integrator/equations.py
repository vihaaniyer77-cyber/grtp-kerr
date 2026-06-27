"""
Right-hand side equations of motion for a charged particle under Lorentz force.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..spacetime.kerr import metric, christoffel
from ..fields.xpoint import FieldProfile

__all__ = ["lorentz_rhs", "normalization_error"]

def lorentz_rhs(
    tau:      float,
    y:        NDArray,
    M:        float,
    a:        float,
    q_over_m: float,
    field:    FieldProfile,
) -> NDArray:
    """Calculate RHS of the 8-ODE system (dx/dtau, du/dtau)."""
    _t, r, theta, _phi = y[0], y[1], y[2], y[3]
    u = y[4:]

    r_plus = M + np.sqrt(max(0.0, M**2 - a**2))
    r_safe = max(r, r_plus + 1e-4)

    # Gravitational acceleration
    Gamma    = christoffel(r_safe, theta, M, a)
    a_grav   = -np.einsum("mab,a,b->m", Gamma, u, u)

    # Electromagnetic acceleration
    if q_over_m != 0.0:
        F_mixed = field.em_tensor_mixed(r_safe, theta)
        a_em    = q_over_m * F_mixed @ u
    else:
        a_em = np.zeros(4)

    return np.concatenate([u, a_grav + a_em])

def normalization_error(
    y: NDArray, M: float, a: float
) -> float:
    """Measure deviation from the g_uv u^u u^v = -1 constraint."""
    r, theta = float(y[1]), float(y[2])
    u        = y[4:]
    g        = metric(r, theta, M, a)
    norm_sq  = float(np.einsum("mn,m,n->", g, u, u))
    return abs(norm_sq + 1.0)

