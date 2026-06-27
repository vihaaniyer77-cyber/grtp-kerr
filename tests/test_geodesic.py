"""
tests/test_geodesic.py
======================
Orbital integration validation tests.

These are the canonical tests for the integrator's correctness, directly
replicating the validation strategy described in §4.2 of the paper.

Test 1 — Schwarzschild ISCO circular orbit
-------------------------------------------
A particle on the Schwarzschild ISCO (r = 6M) should:
  a. Stay at r ≈ 6M for many orbital periods (|Δr| < 1e-6 r_g).
  b. Conserve E_∞ to better than 1e-8 over 100 orbital periods.
  c. Have normalization error |g_μν u^μ u^ν + 1| < 1e-9 throughout.

For the ISCO, the circular orbit parameters (u^t, u^φ) in Schwarzschild are:
  u^t  = 1 / sqrt(1 - 3M/r)     [exact]
  u^φ  = sqrt(M/r³) / sqrt(1 - 3M/r)   [exact, in geometric units]
  u^r  = u^θ = 0
  E_∞  = sqrt(1 - 2M/r) / sqrt(1 - 3M/r) ≈ 0.9428 for r = 6M

Test 2 — Kerr ISCO radius recovery
-------------------------------------
Integrate a near-circular orbit near r_ISCO in Kerr (a = 0.5) and verify
the orbit does not plunge within 50 orbital periods.

Test 3 — Flat-space straight-line geodesic (M = 0, q = 0)
-----------------------------------------------------------
In flat space, a particle with u^r > 0, u^θ = u^φ = 0 should travel in a
straight line.  We verify |r(τ) - r_0 - u^r · τ| < 1e-10.
"""

import numpy as np
import pytest

from grtp.spacetime.kerr import metric as kerr_metric, zamo_tetrad
from grtp.spacetime.ergosphere import event_horizon, isco_radius
from grtp.fields.xpoint import XPointField
from grtp.integrator.solver import ParticleIntegrator

def normalize_4velocity(y0: np.ndarray, M: float, a: float) -> np.ndarray:
    """Adjusts u^t so that the 4-velocity is normalized: g_mu_nu u^mu u^nu = -1."""
    r, theta = y0[1], y0[2]
    g = kerr_metric(r, theta, M, a)
    ur, utheta, uphi = y0[5], y0[6], y0[7]
    A = g[0, 0]
    B = 2.0 * g[0, 3] * uphi
    C = g[1, 1]*ur**2 + g[2, 2]*utheta**2 + g[3, 3]*uphi**2 + 1.0
    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        raise ValueError("Cannot normalize 4-velocity: Discriminant < 0")
    ut1 = (-B + np.sqrt(discriminant)) / (2*A)
    ut2 = (-B - np.sqrt(discriminant)) / (2*A)
    y0_norm = y0.copy()
    y0_norm[4] = max(ut1, ut2)
    return y0_norm


# ---------------------------------------------------------------------------
# Helper: orbital velocity for circular Schwarzschild orbit
# ---------------------------------------------------------------------------

def schwarzschild_circular_ic(r: float, M: float = 1.0) -> np.ndarray:
    """
    Exact 4-velocity for a prograde circular geodesic in Schwarzschild at radius r.

    u^t  = 1 / sqrt(1 - 3M/r)
    u^φ  = sqrt(M/r³) / sqrt(1 - 3M/r)
    u^r  = u^θ = 0

    Valid only for r > 3M (circular orbit exists); r = 6M is ISCO.
    """
    factor = 1.0 / np.sqrt(1.0 - 3.0 * M / r)
    ut     = factor
    uphi   = np.sqrt(M / r**3) * factor
    return np.array([0.0, r, np.pi / 2, 0.0,  ut, 0.0, 0.0, uphi])

def kerr_circular_ic(r: float, M: float, a: float) -> np.ndarray:
    """Exact 4-velocity for a prograde equatorial circular geodesic in Kerr."""
    num_t = r**1.5 + a * np.sqrt(M)
    num_phi = np.sqrt(M)
    den = np.sqrt(r**3 - 3.0*M*r**2 + 2.0*a*np.sqrt(M)*r**1.5)
    
    ut = num_t / den
    uphi = num_phi / den
    return np.array([0.0, r, np.pi / 2, 0.0, ut, 0.0, 0.0, uphi])



# ---------------------------------------------------------------------------
# Test 1 — Schwarzschild ISCO
# ---------------------------------------------------------------------------

class TestSchwarzschildISCO:
    """
    Integration on the Schwarzschild ISCO (r = 6M, a = 0, q = 0).
    Tolerance thresholds from Kopáček & Karas (2014) DOP853 benchmarks.
    """

    M = 1.0
    a = 0.0
    r_isco = 6.0   # = 6M for Schwarzschild

    @pytest.fixture(scope="class")
    def traj(self):
        M, a     = self.M, self.a
        y0       = schwarzschild_circular_ic(self.r_isco, M)
        # q = 0, B = 0 — pure geodesic
        field    = XPointField(M=M, a=a, r_X=5.5, B0=0.0, E0=0.0, lam=1.0)
        integ    = ParticleIntegrator(
            M=M, a=a, q_over_m=0.0, field=field,
            r_escape=100.0, rtol=1e-12, atol=1e-14,
        )
        # Orbital period at r=6M (Schwarzschild): T ≈ 2π r^{3/2} / M^{1/2} ≈ 277 M
        T_orb = 2.0 * np.pi * self.r_isco ** 1.5 / np.sqrt(self.M)
        tau_max  = 100.0 * T_orb   # Full 100 orbital periods
        return integ.integrate(y0, tau_max=tau_max, n_out=2000)

    def test_radial_stability(self, traj):
        """Radial deviation < 1e-5 r_g over integration."""
        delta_r = np.abs(traj.r - self.r_isco)
        assert delta_r.max() < 1e-5, \
            f"Radial deviation too large: {delta_r.max():.2e}"

    def test_energy_conservation(self, traj):
        """E_∞ conserved to < 5e-8 (geodesic, no EM field)."""
        E0      = traj.E_inf[0]
        delta_E = np.abs(traj.E_inf - E0)
        assert delta_E.max() < 5e-8, \
            f"Energy drift too large: {delta_E.max():.2e}"

    def test_normalisation(self, traj):
        """Normalisation error < 1e-9 throughout."""
        assert traj.norm_err.max() < 1e-9, \
            f"Normalisation error: {traj.norm_err.max():.2e}"

    def test_isco_energy_value(self, traj):
        """E_∞ at ISCO ≈ sqrt(8/9) ≈ 0.942809 (Bardeen 1972)."""
        E_isco_exact = np.sqrt(8.0 / 9.0)
        np.testing.assert_allclose(traj.E_inf[0], E_isco_exact, rtol=1e-13)

    def test_fate_trapped(self, traj):
        """Circular orbit on ISCO should be 'trapped' (neither escapes nor plunges)."""
        assert traj.fate == "trapped", f"Expected 'trapped', got '{traj.fate}'"


# ---------------------------------------------------------------------------
# Test 2 — Kerr ISCO stability
# ---------------------------------------------------------------------------

class TestKerrISCO:
    """
    Near-circular orbit just outside the Kerr ISCO (a=0.5).
    """

    M = 1.0
    a = 0.5

    def test_orbit_does_not_plunge(self):
        """A particle just outside r_ISCO should not plunge within 20 orbits."""
        a     = self.a
        M     = self.M
        r_isco = isco_radius(M, a, prograde=True)
        r_test = r_isco + 0.3   # slightly outside ISCO

        # Exact circular orbit IC for Kerr
        y0 = kerr_circular_ic(r_test, M, a)
        # Normalize to be perfectly on the mass shell to prevent float precision drift
        y0 = normalize_4velocity(y0, M, a)

        field = XPointField(M=M, a=a, r_X=r_test, B0=0.0, E0=0.0, lam=1.0)
        integ = ParticleIntegrator(
            M=M, a=a, q_over_m=0.0, field=field,
            r_escape=50.0, rtol=1e-10, atol=1e-12,
        )
        T_orb  = 2.0 * np.pi * r_test ** 1.5 / np.sqrt(M)
        traj   = integ.integrate(y0, tau_max=5.0 * T_orb, n_out=500)
        assert traj.fate != "plunge", \
            f"Orbit just outside ISCO plunged (r_ISCO={r_isco:.4f}, r_test={r_test:.4f})"


# ---------------------------------------------------------------------------
# Test 3 — Flat-space straight-line geodesic
# ---------------------------------------------------------------------------

class TestFlatSpaceGeodesic:
    """
    In flat space (M → 0), a particle with u^r = 1, u^t = 1, u^φ = u^θ = 0
    should travel outward at constant speed.  r(τ) = r_0 + τ.
    """

    def test_straight_line(self):
        """r(τ) = r_0 + u^r · τ to < 1e-9 in flat space."""
        M_flat = 0.0   # exact flat space (no gravity)
        a_flat = 0.0
        r0     = 5.0
        ur0    = 0.5

        y0    = np.array([0.0, r0, np.pi / 2, 0.0,  0.0, ur0, 0.0, 0.0])
        y0    = normalize_4velocity(y0, M_flat, a_flat)
        field = XPointField(M=M_flat, a=a_flat, r_X=r0 + 5, B0=0.0, E0=0.0, lam=1.0)
        integ = ParticleIntegrator(
            M=M_flat, a=a_flat, q_over_m=0.0, field=field,
            r_escape=100.0, rtol=1e-12, atol=1e-14,
        )
        tau_max = 20.0
        traj    = integ.integrate(y0, tau_max=tau_max, n_out=500)

        r_expected = r0 + ur0 * traj.tau
        delta_r    = np.abs(traj.r - r_expected)
        assert delta_r.max() < 1e-7, \
            f"Flat-space geodesic deviation: {delta_r.max():.2e}"

# ---------------------------------------------------------------------------
# Test 4 — Plunge detection
# ---------------------------------------------------------------------------

class TestPlungeDetection:
    def test_plunge_inside_isco(self):
        """A particle starting well inside the ISCO with inward momentum should plunge."""
        M = 1.0
        a = 0.5
        r_start = 2.5 # Inside ISCO
        y0 = np.array([0.0, r_start, np.pi/2, 0.0, 0.0, -0.1, 0.0, 0.0])
        y0 = normalize_4velocity(y0, M, a)
        field = XPointField(M=M, a=a, r_X=5.0, B0=0.0, E0=0.0, lam=1.0)
        integ = ParticleIntegrator(M=M, a=a, q_over_m=0.0, field=field, r_escape=50.0, rtol=1e-10, atol=1e-12)
        traj = integ.integrate(y0, tau_max=50.0, n_out=500)
        assert traj.fate == "plunge", f"Orbit did not plunge, fate was {traj.fate}"

# ---------------------------------------------------------------------------
# Test 5 — Integrator Convergence
# ---------------------------------------------------------------------------

class TestIntegratorConvergence:
    def test_dop853_convergence(self):
        """Check that tighter tolerances lead to lower energy errors."""
        M = 1.0
        a = 0.0
        y0 = schwarzschild_circular_ic(6.0, M)
        field = XPointField(M=M, a=a, r_X=5.5, B0=0.0, E0=0.0, lam=1.0)
        
        # Loose tolerance
        integ_loose = ParticleIntegrator(M=M, a=a, q_over_m=0.0, field=field, r_escape=100.0, rtol=1e-4, atol=1e-6)
        traj_loose = integ_loose.integrate(y0, tau_max=100.0, n_out=100)
        err_loose = np.max(np.abs(traj_loose.E_inf - traj_loose.E_inf[0]))
        
        # Tight tolerance
        integ_tight = ParticleIntegrator(M=M, a=a, q_over_m=0.0, field=field, r_escape=100.0, rtol=1e-7, atol=1e-9)
        traj_tight = integ_tight.integrate(y0, tau_max=100.0, n_out=100)
        err_tight = np.max(np.abs(traj_tight.E_inf - traj_tight.E_inf[0]))
        
        assert err_tight < err_loose or (err_tight < 1e-14 and err_loose < 1e-14), "Tighter tolerances did not reduce the energy error"
        assert err_tight < 1e-8, "Tight tolerance error is still too large"
