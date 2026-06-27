"""
DOP853 integrator wrapper for general-relativistic particle trajectories.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from ..spacetime.kerr import metric as kerr_metric
from ..spacetime.ergosphere import event_horizon
from ..fields.xpoint import FieldProfile
from .equations import lorentz_rhs, normalization_error
from ..fields.potentials import KillingEnergy

__all__ = ["Trajectory", "ParticleIntegrator"]

@dataclass
class Trajectory:
    """Integrated trajectory data."""
    tau:       NDArray
    x:         NDArray
    u:         NDArray
    E_inf:     NDArray
    norm_err:  NDArray
    fate:      str
    success:   bool
    tau_event: Optional[float] = None

    @property
    def r(self) -> NDArray:
        return self.x[:, 1]

    @property
    def theta(self) -> NDArray:
        return self.x[:, 2]

    @property
    def phi(self) -> NDArray:
        return self.x[:, 3]

    @property
    def t_coord(self) -> NDArray:
        return self.x[:, 0]

    @property
    def energy_gain(self) -> float:
        """Fractional energy gain ΔE_∞ / (m c^2). (m = 1 in geometric units)"""
        dE = float(self.E_inf[-1] - self.E_inf[0])
        return dE

    @property
    def crosses_ergosphere(self) -> bool:
        """
        Lightweight proxy: True if any trajectory point has r < 2.1 M.

        This is a conservative heuristic — the true ergosphere boundary is
        r_E(θ) = M + √(M² − a² cos²θ), which varies with spin and polar angle.
        At the equator r_E(π/2) = 2M for all a/M.  For a rigorous per-point
        check, use `grtp.spacetime.ergosphere.is_in_ergosphere(r, θ, M, a)`.

        Note: Trajectory does not store M or a, so the exact test requires
        the caller to supply these parameters externally.
        """
        # 2.1M is a safe upper bound: r_E(θ) ≤ 2M at equator and ≤ r_+ + 2M/Σ
        # for off-equatorial points.  False negatives (missed ergosphere crossings)
        # are possible for high-spin, off-equatorial trajectories near the poles.
        return bool(np.any(self.x[:, 1] < 2.1))

    def summary(self) -> str:
        return (
            f"Trajectory | fate={self.fate:8s} | "
            f"E_∞: {self.E_inf[0]:.5f} → {self.E_inf[-1]:.5f} "
            f"(ΔE/E₀={self.energy_gain:+.3%}) | "
            f"max|norm_err|={self.norm_err.max():.2e} | "
            f"N_steps={len(self.tau)}"
        )

class ParticleIntegrator:
    """Trajectory integrator using SciPy DOP853."""

    def __init__(
        self,
        M:              float,
        a:              float,
        q_over_m:       float,
        field:          FieldProfile,
        r_escape:       float = 100.0,
        rtol:           float = 1e-10,
        atol:           float = 1e-12,
        norm_tol:       float = 1e-7,
        horizon_buffer: float = 5e-3,
    ) -> None:
        self.M              = float(M)
        self.a              = float(a)
        self.q_over_m       = float(q_over_m)
        self.field          = field
        self.r_escape       = float(r_escape)
        self.rtol           = rtol
        self.atol           = atol
        self.norm_tol       = norm_tol
        self.horizon_buffer = horizon_buffer
        self._r_plus        = event_horizon(M, a)

        # Pre-compute KillingEnergy spline for high-precision energy tracking
        self._ke = KillingEnergy(
            M=self.M, a=self.a, q_over_m=self.q_over_m, field=self.field,
            theta_ref=np.pi/2, r_min=self._r_plus + horizon_buffer, r_max=r_escape
        )

    def _make_events(self):
        r_thresh_low  = self._r_plus + self.horizon_buffer
        r_thresh_high = self.r_escape

        def horizon_event(tau: float, y: NDArray) -> float:
            return y[1] - r_thresh_low

        horizon_event.terminal  = True
        horizon_event.direction = -1.0

        def escape_event(tau: float, y: NDArray) -> float:
            return y[1] - r_thresh_high

        escape_event.terminal  = True
        escape_event.direction = +1.0

        return [horizon_event, escape_event]

    def integrate(
        self,
        y0:        NDArray,
        tau_max:   float,
        n_out:     int   = 2000,
        tau_eval:  Optional[NDArray] = None,
    ) -> Trajectory:
        """Integrate a single trajectory."""
        y0 = np.asarray(y0, dtype=float).copy()
        y0 = self._enforce_normalisation(y0)

        r_thresh_low = self._r_plus + self.horizon_buffer
        if y0[1] < r_thresh_low:
            return Trajectory(
                tau       = np.array([0.0]),
                x         = np.array([y0[:4]]),
                u         = np.array([y0[4:]]),
                E_inf     = np.array([self._ke(y0[1], y0[2], y0[4], y0[7])]),
                norm_err  = np.array([normalization_error(y0, self.M, self.a)]),
                fate      = "plunge",
                success   = True,
                tau_event = 0.0,
            )
        if y0[1] > self.r_escape:
            return Trajectory(
                tau       = np.array([0.0]),
                x         = np.array([y0[:4]]),
                u         = np.array([y0[4:]]),
                E_inf     = np.array([self._ke(y0[1], y0[2], y0[4], y0[7])]),
                norm_err  = np.array([normalization_error(y0, self.M, self.a)]),
                fate      = "escape",
                success   = True,
                tau_event = 0.0,
            )

        if tau_eval is None:
            tau_eval = np.linspace(0.0, tau_max, n_out)
        tau_span = (0.0, float(tau_max))
        events = self._make_events()

        M, a, q_over_m, field = self.M, self.a, self.q_over_m, self.field

        def rhs(tau: float, y: NDArray) -> NDArray:
            return lorentz_rhs(tau, y, M, a, q_over_m, field)

        sol = solve_ivp(
            rhs,
            tau_span,
            y0,
            method       = "DOP853",
            t_eval       = tau_eval,
            events       = events,
            rtol         = self.rtol,
            atol         = self.atol,
            dense_output = False,
            max_step     = tau_max / 10.0,
        )

        tau_event = None
        if sol.t_events[0].size > 0:
            fate      = "plunge"
            tau_event = float(sol.t_events[0][0])
        elif sol.t_events[1].size > 0:
            fate      = "escape"
            tau_event = float(sol.t_events[1][0])
        else:
            fate = "trapped"

        if not sol.success:
            warnings.warn(
                f"DOP853 did not converge: {sol.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        tau_arr = sol.t
        x_arr   = sol.y[:4].T
        u_arr   = sol.y[4:].T
        N       = tau_arr.shape[0]

        E_inf    = np.zeros(N)
        norm_err = np.zeros(N)

        for i in range(N):
            yi          = sol.y[:, i]
            E_inf[i]    = self._ke(float(yi[1]), float(yi[2]), float(yi[4]), float(yi[7]))
            norm_err[i] = normalization_error(yi, M, a)

        max_norm_err = float(norm_err.max())
        if max_norm_err > self.norm_tol:
            warnings.warn(
                f"Normalisation error peaked at {max_norm_err:.2e} "
                f"(tolerance {self.norm_tol:.1e}). "
                "Consider tightening rtol/atol.",
                UserWarning,
                stacklevel=2,
            )

        return Trajectory(
            tau       = tau_arr,
            x         = x_arr,
            u         = u_arr,
            E_inf     = E_inf,
            norm_err  = norm_err,
            fate      = fate,
            success   = sol.success,
            tau_event = tau_event,
        )

    def integrate_batch(
        self,
        y0_list:  list[NDArray],
        tau_max:  float,
        n_out:    int = 500,
        n_jobs:   int = -1,
    ) -> list[Trajectory]:
        """Integrate a batch of trajectories in parallel."""
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        n_jobs = min(n_jobs, len(y0_list))

        args = [
            (y0, tau_max, n_out,
             self.M, self.a, self.q_over_m, self.field,
             self.r_escape, self.rtol, self.atol,
             self.norm_tol, self.horizon_buffer)
            for y0 in y0_list
        ]

        results: dict[int, Trajectory] = {}
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_integrate_single, arg): idx
                for idx, arg in enumerate(args)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Integrating particles"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    warnings.warn(
                        f"Trajectory {idx} failed: {exc}", RuntimeWarning
                    )
                    results[idx] = None

        return [results[i] for i in range(len(y0_list))]

    def _enforce_normalisation(self, y: NDArray) -> NDArray:
        """Enforce normalisation constraint g_uv u^u u^v = -1."""
        r, theta = float(y[1]), float(y[2])
        g = kerr_metric(r, theta, self.M, self.a)

        ur, utheta, uphi = float(y[5]), float(y[6]), float(y[7])

        S = (
            g[1, 1] * ur    * ur
            + g[2, 2] * utheta * utheta
            + g[3, 3] * uphi   * uphi
        )

        A_coeff =  g[0, 0]
        B_coeff =  2.0 * g[0, 3] * uphi
        C_coeff =  S + 1.0

        discriminant = B_coeff * B_coeff - 4.0 * A_coeff * C_coeff
        if discriminant < 0:
            raise ValueError(
                "Cannot normalise 4-velocity: discriminant < 0."
            )

        ut = (-B_coeff - np.sqrt(discriminant)) / (2.0 * A_coeff)

        y_out    = y.copy()
        y_out[4] = ut
        return y_out

def _integrate_single(args: tuple) -> Trajectory:
    (y0, tau_max, n_out,
     M, a, q_over_m, field,
     r_escape, rtol, atol,
     norm_tol, horizon_buffer) = args

    integrator = ParticleIntegrator(
        M              = M,
        a              = a,
        q_over_m       = q_over_m,
        field          = field,
        r_escape       = r_escape,
        rtol           = rtol,
        atol           = atol,
        norm_tol       = norm_tol,
        horizon_buffer = horizon_buffer,
    )
    return integrator.integrate(y0, tau_max, n_out=n_out)
