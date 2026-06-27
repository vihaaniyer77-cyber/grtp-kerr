"""Boundary definitions for the ergosphere and event horizon."""
from __future__ import annotations

import numpy as np

__all__ = [
    "event_horizon",
    "inner_horizon",
    "ergosphere_radius",
    "isco_radius",
    "is_in_ergosphere",
    "is_outside_horizon",
]


def event_horizon(M: float, a: float) -> float:
    """
    Outer event horizon radius:  r_+ = M + √(M² − a²).

    For an extremal black hole (a = M), r_+ = M.
    For a Schwarzschild black hole (a = 0), r_+ = 2M.

    Parameters
    ----------
    M : float   Black hole mass
    a : float   Spin parameter  |a| ≤ M

    Returns
    -------
    r_plus : float
    """
    discriminant = M * M - a * a
    if discriminant < 0.0:
        raise ValueError(
            f"Spin parameter |a|={abs(a):.4f} exceeds mass M={M:.4f}. "
            "Naked singularity — not supported."
        )
    return M + np.sqrt(discriminant)


def inner_horizon(M: float, a: float) -> float:
    """
    Inner (Cauchy) horizon radius:  r_- = M − √(M² − a²).

    Relevant for the global causal structure; particles reaching r < r_- are
    inside the Cauchy horizon.  Not directly used in ergosphere dynamics.

    Returns
    -------
    r_minus : float
    """
    discriminant = M * M - a * a
    if discriminant < 0.0:
        raise ValueError("Spin |a| > M — naked singularity not supported.")
    return M - np.sqrt(discriminant)


def ergosphere_radius(theta: float, M: float, a: float) -> float:
    """
    Ergosphere (static limit) radius at polar angle θ:

      r_E(θ) = M + √(M² − a² cos²θ)

    At the equator (θ = π/2): r_E = M + M = 2M  (for all a).
    At the poles  (θ = 0, π): r_E = r_+ (ergosphere coincides with horizon).

    Parameters
    ----------
    theta : float   Polar angle [rad]
    M, a  : float   Mass and spin

    Returns
    -------
    r_ergo : float
    """
    cos_t = np.cos(theta)
    return M + np.sqrt(M * M - a * a * cos_t * cos_t)


def is_in_ergosphere(
    r: float, theta: float, M: float, a: float, buffer: float = 1e-4
) -> bool:
    """
    Test whether a point is inside the ergosphere but outside the event horizon.

    Returns True if  r_+ + buffer < r < r_E(θ).

    The `buffer` parameter keeps the test particle safely away from the
    horizon coordinate singularity in BL coordinates.
    """
    r_plus = event_horizon(M, a)
    r_ergo = ergosphere_radius(theta, M, a)
    return (r_plus + buffer) < r < r_ergo


def is_outside_horizon(
    r: float, M: float, a: float, buffer: float = 1e-4
) -> bool:
    """
    Test whether a point is outside the event horizon.

    Returns True if  r > r_+ + buffer.

    Parameters
    ----------
    buffer : float   Safety margin in gravitational radii (default 0.0001 r_g).
    """
    return r > (event_horizon(M, a) + buffer)


def isco_radius(M: float, a: float, prograde: bool = True) -> float:
    """
    Innermost Stable Circular Orbit (ISCO) radius.

    Uses the exact analytical formula from Bardeen, Press & Teukolsky (1972),
    equations (2.21).  This is the standard validation target for geodesic
    integrators (see tests/test_geodesic.py).

    Parameters
    ----------
    M        : float   Black hole mass
    a        : float   Spin parameter  |a| ≤ M
    prograde : bool    True  → prograde (co-rotating) orbit  [smaller r_ISCO]
                       False → retrograde orbit               [larger  r_ISCO]

    Returns
    -------
    r_isco : float

    Examples
    --------
    >>> isco_radius(1.0, 0.0)        # Schwarzschild: 6M
    6.0
    >>> isco_radius(1.0, 1.0)        # Extremal prograde: ~1M (ergosphere edge)
    1.0
    >>> isco_radius(1.0, 1.0, False) # Extremal retrograde: 9M
    9.0
    """
    sign  = -1.0 if prograde else +1.0   # sign convention for orbit direction
    a_hat = a / M                        # dimensionless spin  0 ≤ a/M ≤ 1

    # Equation (2.21) of BPT (1972):
    Z1 = 1.0 + (1.0 - a_hat * a_hat) ** (1.0 / 3.0) * (
        (1.0 + a_hat) ** (1.0 / 3.0) + (1.0 - a_hat) ** (1.0 / 3.0)
    )
    Z2 = np.sqrt(3.0 * a_hat * a_hat + Z1 * Z1)

    return M * (3.0 + Z2 + sign * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))


def ergosphere_width(theta: float, M: float, a: float) -> float:
    """
    Radial width of the ergosphere at polar angle θ:

      Δr_ergo(θ) = r_E(θ) − r_+

    Useful for setting the X-point location and current sheet half-width λ
    relative to the ergosphere geometry.

    Returns
    -------
    width : float   [gravitational radii]
    """
    return ergosphere_radius(theta, M, a) - event_horizon(M, a)
