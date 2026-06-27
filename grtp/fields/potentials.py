"""Electromagnetic 4-potential calculations."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .xpoint import FieldProfile
from ..spacetime.kerr import metric as kerr_metric

__all__ = [
    "em_potential_t",
    "em_potential_phi",
    "build_potential_spline",
    "build_potential_spline_phi",
    "KillingEnergy",
]


# ---------------------------------------------------------------------------
# Internal: covariant F_tr and F_tφ from the field object
# ---------------------------------------------------------------------------

def _F_covariant(
    r: float, theta: float, field: FieldProfile
) -> NDArray:
    """
    Full covariant EM tensor F_αβ = g_αμ g_βν F^μν at (r, θ).

    Returns
    -------
    F_low : (4, 4) ndarray  [F_αβ, fully covariant]
    """
    g    = kerr_metric(r, theta, field.M, field.a)    # g_μν
    Fup  = field.em_tensor(r, theta)                  # F^μν
    # F_αβ = Σ_{μ,ν} g_{αμ} g_{βν} F^{μν}
    return np.einsum("am,bn,mn->ab", g, g, Fup)


# ---------------------------------------------------------------------------
# 1.  One-off potential evaluation (Simpson's rule)
# ---------------------------------------------------------------------------

def em_potential_t(
    r:       float,
    theta:   float,
    field:   FieldProfile,
    r_ref:   float = 100.0,
    n_steps: int   = 400,
) -> float:
    """
    Compute A_t(r, θ) by integrating F_{tr} radially from r_ref to r.

    Formula (Coulomb gauge, static field):
      A_t(r, θ) = −∫_{r_ref}^{r} F_{tr}(r′, θ) dr′

    The integrand F_{tr}(r′, θ) is the (t, r) component of the covariant
    EM tensor, which vanishes exponentially far from the X-point.

    Parameters
    ----------
    r       : float   Target radius
    theta   : float   Polar angle (held fixed; we integrate along constant θ)
    field   : FieldProfile  EM field object
    r_ref   : float   Reference radius where A_t = 0  (should be ≫ r_X)
    n_steps : int     Simpson integration steps (must be even; default 400)

    Returns
    -------
    A_t : float
    """
    if n_steps % 2 != 0:
        n_steps += 1

    r_arr  = np.linspace(r_ref, r, n_steps + 1)
    f_vals = np.array([
        _F_covariant(ri, theta, field)[0, 1]   # F_tr component
        for ri in r_arr
    ])

    # Composite Simpson's rule: ∫ ≈ (h/3)[f₀ + 4f₁ + 2f₂ + 4f₃ + … + fₙ]
    h     = (r - r_ref) / n_steps
    coeff = np.ones(n_steps + 1)
    coeff[1:-1:2] = 4.0  # odd interior points
    coeff[2:-2:2] = 2.0  # even interior points
    A_t   = -(h / 3.0) * np.dot(coeff, f_vals)
    return float(A_t)


def em_potential_phi(
    r:       float,
    theta:   float,
    field:   FieldProfile,
    r_ref:   float = 100.0,
    n_steps: int   = 400,
) -> float:
    """
    Compute A_φ(r, θ) by integrating F_{rφ} radially from r_ref to r.

    Formula (static + axisymmetric field, ∂_φ A_r = 0):
      F_{rφ} = ∂_r A_φ  ⟹  A_φ(r) = ∫_{r_ref}^{r} F_{rφ}(r′, θ) dr′

    A_φ appears in the conserved orbital angular momentum:
      L_∞ = g_{φt} u^t + g_{φφ} u^φ + (q/m) A_φ

    Parameters
    ----------
    Same as em_potential_t.

    Returns
    -------
    A_phi : float
    """
    if n_steps % 2 != 0:
        n_steps += 1

    r_arr  = np.linspace(r_ref, r, n_steps + 1)
    f_vals = np.array([
        _F_covariant(ri, theta, field)[1, 3]   # F_rφ component
        for ri in r_arr
    ])

    h     = (r - r_ref) / n_steps
    coeff = np.ones(n_steps + 1)
    coeff[1:-1:2] = 4.0
    coeff[2:-2:2] = 2.0
    A_phi = (h / 3.0) * np.dot(coeff, f_vals)
    return float(A_phi)


# ---------------------------------------------------------------------------
# 2.  Pre-computed spline for hot-loop use
# ---------------------------------------------------------------------------

def build_potential_spline(
    theta:  float,
    field:  FieldProfile,
    r_min:  float,
    r_max:  float,
    n_grid: int = 5000,
) -> CubicSpline:
    """
    Pre-compute A_t on a radial grid and return a fast CubicSpline interpolant.

    Calling this once before the integrator loop and then evaluating
    `spline(r)` at each step is ~100× faster than recomputing the integral.

    The integral is built cumulatively via the trapezoid rule starting from
    r_max (where A_t = 0) and stepping inward.

    Parameters
    ----------
    theta   : float   Polar angle at which the spline is constructed
    field   : FieldProfile
    r_min   : float   Minimum radius on the grid (should be > r_+)
    r_max   : float   Maximum radius (reference, A_t = 0 here)
    n_grid  : int     Number of grid points

    Returns
    -------
    spline : CubicSpline   callable  r ↦ A_t(r, theta)
    """
    # Build grid from r_max inward to r_min
    r_grid    = np.linspace(r_max, r_min, n_grid)
    A_vals    = np.zeros(n_grid)
    A_vals[0] = 0.0   # boundary condition: A_t = 0 at r_max

    for i in range(1, n_grid):
        r0 = r_grid[i - 1]
        r1 = r_grid[i]             # r1 < r0  (stepping inward)
        F0 = _F_covariant(r0, theta, field)[0, 1]   # F_tr at r0
        F1 = _F_covariant(r1, theta, field)[0, 1]   # F_tr at r1
        dr = r1 - r0               # negative (inward step)
        # A_t(r1) = A_t(r0) - ∫_{r0}^{r1} F_tr dr  (trapezoid step)
        A_vals[i] = A_vals[i - 1] - 0.5 * (F0 + F1) * dr

    # Return spline on increasing-r grid
    return CubicSpline(r_grid[::-1], A_vals[::-1])


def build_potential_spline_phi(
    theta:  float,
    field:  FieldProfile,
    r_min:  float,
    r_max:  float,
    n_grid: int = 5000,
) -> CubicSpline:
    """
    Pre-compute A_phi on a radial grid and return a fast CubicSpline interpolant.
    """
    r_grid    = np.linspace(r_max, r_min, n_grid)
    A_vals    = np.zeros(n_grid)
    A_vals[0] = 0.0   # boundary condition: A_phi = 0 at r_max

    for i in range(1, n_grid):
        r0 = r_grid[i - 1]
        r1 = r_grid[i]
        F0 = _F_covariant(r0, theta, field)[1, 3]   # F_rphi at r0
        F1 = _F_covariant(r1, theta, field)[1, 3]   # F_rphi at r1
        dr = r1 - r0
        A_vals[i] = A_vals[i - 1] + 0.5 * (F0 + F1) * dr

    return CubicSpline(r_grid[::-1], A_vals[::-1])


# ---------------------------------------------------------------------------
# 3.  Killing energy helper
# ---------------------------------------------------------------------------

class KillingEnergy:
    """
    Computes the conserved Killing energy E_∞ along a trajectory.

    E_∞ = −(g_tt u^t + g_tφ u^φ) − (q/m) A_t(r, θ)

    For a purely geodesic (uncharged) particle, the −(q/m)A_t term vanishes.
    For charged particles in the reconnection field, A_t is pre-computed
    on a CubicSpline grid for efficiency.

    Usage
    -----
    >>> ke = KillingEnergy(M=1.0, a=0.9, q_over_m=1.0, field=field,
    ...                    theta_ref=np.pi/2, r_min=1.5, r_max=80.0)
    >>> E_inf = ke(r=2.3, theta=np.pi/2, ut=2.1, uphi=0.3)
    """

    def __init__(
        self,
        M:         float,
        a:         float,
        q_over_m:  float,
        field:     FieldProfile,
        theta_ref: float  = np.pi / 2,
        r_min:     float  = None,
        r_max:     float  = 100.0,
        n_grid:    int    = 5000,
    ) -> None:
        self.M        = M
        self.a        = a
        self.q_over_m = q_over_m
        self.field    = field

        # Pre-compute A_t and A_phi splines at theta_ref (equatorial plane by default)
        from ..spacetime.ergosphere import event_horizon
        if r_min is None:
            r_min = event_horizon(M, a) + 0.05
        self._spline    = build_potential_spline(
            theta_ref, field, r_min, r_max, n_grid
        )
        self._spline_phi = build_potential_spline_phi(
            theta_ref, field, r_min, r_max, n_grid
        )
        self._r_min     = r_min
        self._r_max     = r_max

    def get_A_t(self, r: float) -> float:
        r_clamped = float(np.clip(r, self._r_min, self._r_max))
        return float(self._spline(r_clamped))

    def get_A_phi(self, r: float) -> float:
        r_clamped = float(np.clip(r, self._r_min, self._r_max))
        return float(self._spline_phi(r_clamped))

    def __call__(
        self,
        r:     float,
        theta: float,
        ut:    float,
        uphi:  float,
    ) -> float:
        """
        Evaluate E_∞ at a single phase-space point.

        Parameters
        ----------
        r, theta : float   BL position
        ut, uphi : float   Contravariant 4-velocity components u^t, u^φ

        Returns
        -------
        E_inf : float   Conserved Killing energy (negative = bound)
        """
        g = kerr_metric(r, theta, self.M, self.a)
        # Purely gravitational contribution
        E_grav = -(g[0, 0] * ut + g[0, 3] * uphi)
        # EM contribution (evaluate A_t from spline within valid range)
        A_t       = self.get_A_t(r)
        return float(E_grav - self.q_over_m * A_t)

    def batch(
        self,
        x_arr: NDArray,
        u_arr: NDArray,
    ) -> NDArray:
        """
        Vectorised E_∞ computation over a trajectory array.

        Parameters
        ----------
        x_arr : (N, 4) ndarray   positions  x^μ
        u_arr : (N, 4) ndarray   4-velocities u^μ

        Returns
        -------
        E_inf : (N,) ndarray
        """
        N     = x_arr.shape[0]
        E_inf = np.zeros(N)
        for i in range(N):
            r_i     = float(x_arr[i, 1])
            theta_i = float(x_arr[i, 2])
            ut_i    = float(u_arr[i, 0])
            uphi_i  = float(u_arr[i, 3])
            E_inf[i] = self(r_i, theta_i, ut_i, uphi_i)
        return E_inf
