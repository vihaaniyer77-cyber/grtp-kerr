"""
grtp/analysis/energy.py
=======================
Conserved quantities along test-particle trajectories in Kerr spacetime.

The Kerr metric admits two Killing vectors (∂_t, ∂_φ), yielding two conserved
quantities for *geodesic* motion (q = 0):

  E_∞  = −p_t = −(g_tt u^t + g_tφ u^φ)          [Killing energy at infinity]
  L    =  p_φ =  g_φt u^t + g_φφ u^φ             [orbital angular momentum]

For charged particles in the reconnection EM field, E_∞ and L receive
electromagnetic corrections:

  E_∞  = −(g_tt u^t + g_tφ u^φ) − (q/m) A_t
  L    =  (g_φt u^t + g_φφ u^φ) + (q/m) A_φ

There is a third conserved quantity for neutral geodesics — the **Carter
constant** Q — arising from a hidden symmetry (rank-2 Killing tensor).  It is
NOT conserved for charged particles in a non-trivial EM field.  We include it
as a diagnostic: its drift along a trajectory measures how much the EM field
breaks the hidden symmetry.

Energy interpretation
---------------------
  E_∞ > 0   : unbound (can reach infinity with kinetic energy E_∞ − 1)
  E_∞ = 0   : marginally bound (zero energy at infinity)
  E_∞ < 0   : negative energy (inside ergosphere; drives the Penrose process)
  E_∞ < −1  : deeply bound (rare; requires strong EM acceleration)

The magnetic Penrose process: a particle with E_∞ < 0 plunges into the black
hole, reducing its spin angular momentum.  The ejected (escape-fate) companion
carries away E_∞ > E_initial, extracting rotational energy.

References
----------
  Carter (1968), Phys. Rev. 174, 1559  — Carter constant
  Bardeen, Press & Teukolsky (1972), ApJ 178, 347
  Comisso & Asenjo (2021), PRL 127, 111101
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..spacetime.kerr import metric as kerr_metric

__all__ = [
    "killing_energy",
    "angular_momentum",
    "carter_constant",
    "EnergyTracker",
]


# ---------------------------------------------------------------------------
# Point-wise conserved quantities
# ---------------------------------------------------------------------------

def killing_energy(
    r:        float,
    theta:    float,
    M:        float,
    a:        float,
    u:        NDArray,
    q_over_m: float = 0.0,
    A_t:      float = 0.0,
) -> float:
    """
    Conserved Killing energy at infinity:

      E_∞ = −(g_tt u^t + g_tφ u^φ) − (q/m) A_t

    Parameters
    ----------
    r, theta  : float         BL position
    M, a      : float         Spacetime parameters
    u         : (4,) ndarray  Contravariant 4-velocity (u^t, u^r, u^θ, u^φ)
    q_over_m  : float         Charge-to-mass ratio  (0 → geodesic)
    A_t       : float         EM potential A_t at this point

    Returns
    -------
    E_inf : float
    """
    g = kerr_metric(r, theta, M, a)
    return float(-(g[0, 0] * u[0] + g[0, 3] * u[3]) - q_over_m * A_t)


def angular_momentum(
    r:        float,
    theta:    float,
    M:        float,
    a:        float,
    u:        NDArray,
    q_over_m: float = 0.0,
    A_phi:    float = 0.0,
) -> float:
    """
    Conserved z-component of orbital angular momentum:

      L = g_φt u^t + g_φφ u^φ + (q/m) A_φ

    Returns
    -------
    L : float  (positive = prograde, negative = retrograde)
    """
    g = kerr_metric(r, theta, M, a)
    return float(g[3, 0] * u[0] + g[3, 3] * u[3] + q_over_m * A_phi)


def carter_constant(
    r:     float,
    theta: float,
    M:     float,
    a:     float,
    u:     NDArray,
    E:     Optional[float] = None,
    L:     Optional[float] = None,
) -> float:
    """
    Carter constant Q for a massive particle on a geodesic in Kerr.

    Definition (Carter 1968, Eq. 4.14; normalised for m = 1):

      Q = Σ² (u^θ)² + cos²θ [a²(1 − E²) + (L/sinθ)²]

    where E and L are the Killing energy and angular momentum.

    .. warning::
       Q is conserved **only for geodesics** (q = 0).  In the presence of the
       reconnection EM field, use its drift dQ/dτ as a diagnostic of how
       strongly the EM force breaks the hidden Kerr symmetry.

    Parameters
    ----------
    r, theta : float         BL position
    M, a     : float         Spacetime parameters
    u        : (4,) ndarray  Contravariant 4-velocity
    E, L     : float, opt    Pre-computed Killing energy and angular momentum.
                             If None, computed internally.

    Returns
    -------
    Q : float
    """
    if E is None:
        E = killing_energy(r, theta, M, a, u)
    if L is None:
        L = angular_momentum(r, theta, M, a, u)

    g     = kerr_metric(r, theta, M, a)
    Sigma = r * r + a * a * np.cos(theta) ** 2
    sin_t = np.sin(theta)

    # p_θ = g_θθ u^θ = Σ u^θ
    p_theta = g[2, 2] * u[2]   # = Σ u^θ

    sin2 = sin_t * sin_t
    cos2 = np.cos(theta) ** 2

    # Carter constant:  Q = (Σ u^θ)² + cos²θ [a²(1 − E²) + L²/sin²θ]
    if sin2 > 1e-15:
        Q = p_theta * p_theta + cos2 * (a * a * (1.0 - E * E) + L * L / sin2)
    else:
        # At poles: L must be 0 for non-singular motion
        Q = p_theta * p_theta

    return float(Q)


# ---------------------------------------------------------------------------
# Trajectory-level energy tracker
# ---------------------------------------------------------------------------

@dataclass
class EnergyRecord:
    """Snapshot of all conserved quantities at a single proper-time step."""
    tau:   float
    r:     float
    theta: float
    E_inf: float
    L:     float
    Q:     float       # Carter constant (conserved only for geodesics)
    norm:  float       # |g_μν u^μ u^ν + 1|  (normalisation error)


class EnergyTracker:
    """
    Computes and stores the full suite of conserved quantities along a
    trajectory.

    Supports both post-hoc analysis (from a `Trajectory` object) and
    optional pre-computed EM potential corrections via a `KillingEnergy`
    instance.

    Usage
    -----
    >>> from grtp.integrator.solver import ParticleIntegrator
    >>> from grtp.analysis.energy import EnergyTracker
    >>>
    >>> traj = integrator.integrate(y0, tau_max=500)
    >>> tracker = EnergyTracker(M=1.0, a=0.9, q_over_m=1.0)
    >>> records = tracker.from_trajectory(traj)
    >>> print(f"Max Carter drift: {tracker.carter_drift(records):.2e}")
    """

    def __init__(
        self,
        M:         float,
        a:         float,
        q_over_m:  float = 0.0,
        killing_energy_obj=None,   # KillingEnergy instance for A_t correction
    ) -> None:
        self.M               = M
        self.a               = a
        self.q_over_m        = q_over_m
        self._ke_obj         = killing_energy_obj

    def from_trajectory(self, traj) -> list[EnergyRecord]:
        """
        Compute conserved quantities at each output point of a Trajectory.

        Parameters
        ----------
        traj : Trajectory   Output of ParticleIntegrator.integrate()

        Returns
        -------
        records : list[EnergyRecord]
        """
        from ..integrator.equations import normalization_error

        records = []
        for i in range(len(traj.tau)):
            r_i     = float(traj.x[i, 1])
            theta_i = float(traj.x[i, 2])
            u_i     = traj.u[i]
            y_i     = np.concatenate([traj.x[i], traj.u[i]])

            # EM potential correction
            if self._ke_obj is not None:
                A_t = self._ke_obj.get_A_t(r_i)
                A_phi = self._ke_obj.get_A_phi(r_i)
            else:
                A_t = 0.0
                A_phi = 0.0

            E  = killing_energy(r_i, theta_i, self.M, self.a, u_i,
                                self.q_over_m, A_t)
            Lz = angular_momentum(r_i, theta_i, self.M, self.a, u_i,
                                  self.q_over_m, A_phi)
            Q  = carter_constant(r_i, theta_i, self.M, self.a, u_i,
                                 E=E, L=Lz)
            ne = normalization_error(y_i, self.M, self.a)

            records.append(EnergyRecord(
                tau=float(traj.tau[i]),
                r=r_i, theta=theta_i,
                E_inf=E, L=Lz, Q=Q, norm=ne,
            ))
        return records

    def carter_drift(self, records: list[EnergyRecord]) -> float:
        """
        Max absolute change in Carter constant Q relative to initial value.
        For a pure geodesic this should be ≲ 1e-8.  Large values indicate
        strong EM-force breaking of the hidden Kerr symmetry.
        """
        Q0 = records[0].Q
        Qs = np.array([rec.Q for rec in records])
        if abs(Q0) > 1e-12:
            return float(np.max(np.abs((Qs - Q0) / Q0)))
        return float(np.max(np.abs(Qs - Q0)))

    def energy_variance(self, records: list[EnergyRecord]) -> float:
        """
        Std dev of E_∞ along the trajectory.  For a purely magnetic field
        (E = 0) with the correct gauge, this should be ≲ 1e-8.
        """
        Es = np.array([rec.E_inf for rec in records])
        return float(np.std(Es))
