"""
tests/test_kerr.py
==================
Unit tests for the Kerr spacetime geometry module.

Validation targets
------------------
  1. Metric signature: g_tt < 0 outside ergosphere, > 0 inside.
  2. Inverse metric identity: g * g_inv = I to machine precision.
  3. Christoffel symmetry: Γ^μ_αβ = Γ^μ_βα.
  4. ZAMO tetrad orthonormality: η_{ab} = g_{μν} e^μ_(a) e^ν_(b).
  5. Ergosphere radius: r_E(π/2) = 2M for all a.
  6. ISCO: Schwarzschild r_ISCO = 6M; BPT (1972) extremal values.
  7. Frame-dragging: ω > 0 inside ergosphere for prograde spin.
"""

import numpy as np
import pytest

from grtp.spacetime.kerr import (
    kerr_scalars, metric, inverse_metric, christoffel,
    zamo_tetrad, frame_drag_omega, lapse_function,
)
from grtp.spacetime.ergosphere import (
    event_horizon, ergosphere_radius, isco_radius,
    is_in_ergosphere, is_outside_horizon,
)


# ---------------------------------------------------------------------------
# 1. Metric signature
# ---------------------------------------------------------------------------

class TestMetricSignature:
    def test_gtt_negative_outside_ergosphere(self):
        """g_tt < 0 outside the static limit (particle can be at rest)."""
        M, a = 1.0, 0.9
        r_ergo = ergosphere_radius(np.pi / 2, M, a)
        g = metric(r_ergo + 0.5, np.pi / 2, M, a)
        assert g[0, 0] < 0.0, f"g_tt={g[0,0]} should be < 0 outside ergosphere"

    def test_gtt_positive_inside_ergosphere(self):
        """g_tt > 0 inside the ergosphere (Killing ∂_t is spacelike there)."""
        M, a   = 1.0, 0.9
        r_plus = event_horizon(M, a)
        r_ergo = ergosphere_radius(np.pi / 2, M, a)
        r_mid  = 0.5 * (r_plus + r_ergo)
        g = metric(r_mid, np.pi / 2, M, a)
        assert g[0, 0] > 0.0, f"g_tt={g[0,0]} should be > 0 inside ergosphere"

    def test_metric_symmetry(self):
        """g_μν = g_νμ."""
        g = metric(3.5, np.pi / 3, M=1.0, a=0.7)
        np.testing.assert_allclose(g, g.T, atol=1e-15)

    def test_schwarzschild_gtt(self):
        """g_tt = -(1 - 2M/r) for a = 0 (Schwarzschild)."""
        M, a, r = 1.0, 0.0, 10.0
        g = metric(r, np.pi / 2, M, a)
        expected = -(1.0 - 2.0 * M / r)
        np.testing.assert_allclose(g[0, 0], expected, rtol=1e-14)

    def test_schwarzschild_grr(self):
        """g_rr = (1 - 2M/r)^{-1} for a = 0."""
        M, a, r = 1.0, 0.0, 10.0
        g = metric(r, np.pi / 2, M, a)
        expected = 1.0 / (1.0 - 2.0 * M / r)
        np.testing.assert_allclose(g[1, 1], expected, rtol=1e-14)

    def test_flat_limit(self):
        """g_μν → diag(-1,1,r²,r²sin²θ) at large r (M → 0 limit)."""
        r, theta = 100.0, np.pi / 3
        g = metric(r, theta, M=0.001, a=0.0)
        expected_diag = np.array([-1.0, 1.0, r**2, r**2 * np.sin(theta)**2])
        np.testing.assert_allclose(np.diag(g), expected_diag, rtol=1e-3)


# ---------------------------------------------------------------------------
# 2. Inverse metric identity
# ---------------------------------------------------------------------------

class TestInverseMetric:
    @pytest.mark.parametrize("r, theta, a", [
        (5.0, np.pi / 2, 0.0),    # Schwarzschild
        (2.0, np.pi / 2, 0.9),    # Inside ergosphere
        (4.0, np.pi / 4, 0.5),    # Off-equator, mid-spin
        (1.8, np.pi / 3, 0.99),   # Near-extremal
    ])
    def test_g_ginv_identity(self, r, theta, a):
        """g * g^{-1} = I to within 1e-13."""
        M  = 1.0
        g    = metric(r, theta, M, a)
        ginv = inverse_metric(r, theta, M, a)
        prod = g @ ginv
        np.testing.assert_allclose(prod, np.eye(4), atol=1e-13,
                                   err_msg=f"g*g_inv ≠ I at r={r}, θ={theta:.2f}, a={a}")


# ---------------------------------------------------------------------------
# 3. Christoffel symmetry
# ---------------------------------------------------------------------------

class TestChristoffel:
    def test_lower_index_symmetry(self):
        """Γ^μ_αβ = Γ^μ_βα (metric connection has no torsion)."""
        M, a = 1.0, 0.9
        G = christoffel(3.0, np.pi / 2, M, a)
        diff = np.max(np.abs(G - G.transpose(0, 2, 1)))
        assert diff < 1e-12, f"Christoffel symmetry violation: {diff:.2e}"

    def test_schwarzschild_Gamma_t_tr(self):
        """
        In Schwarzschild: Γ^t_tr = M / (r(r - 2M)).
        """
        M, a = 1.0, 0.0
        r    = 5.0
        G    = christoffel(r, np.pi / 2, M, a)
        expected = M / (r * (r - 2.0 * M))
        # Γ^t_tr = Γ[0, 0, 1]
        np.testing.assert_allclose(G[0, 0, 1], expected, rtol=1e-6,
                                   err_msg=f"Γ^t_tr: {G[0,0,1]:.8f} vs {expected:.8f}")


# ---------------------------------------------------------------------------
# 4. ZAMO tetrad orthonormality
# ---------------------------------------------------------------------------

class TestZAMOTetrad:
    @pytest.mark.parametrize("r, theta, a", [
        (2.0, np.pi / 2, 0.9),
        (3.0, np.pi / 3, 0.5),
        (1.6, np.pi / 2, 0.99),
        (5.0, np.pi / 4, 0.0),
    ])
    def test_orthonormality(self, r, theta, a):
        """η_{ab} = g_{μν} e^μ_(a) e^ν_(b) = diag(-1,+1,+1,+1)."""
        M   = 1.0
        tet = zamo_tetrad(r, theta, M, a)
        g   = metric(r, theta, M, a)
        eta = np.einsum("am,bn,mn->ab", tet, tet, g)
        np.testing.assert_allclose(np.diag(eta), [-1, 1, 1, 1], atol=1e-12,
                                   err_msg=f"Tetrad diagonal wrong at r={r}, a={a}")
        off_diag = eta - np.diag(np.diag(eta))
        assert np.max(np.abs(off_diag)) < 1e-12, \
            f"Tetrad off-diagonal non-zero: {np.max(np.abs(off_diag)):.2e}"


# ---------------------------------------------------------------------------
# 5 & 6. Ergosphere and ISCO
# ---------------------------------------------------------------------------

class TestErgosphere:
    def test_ergosphere_equatorial(self):
        """r_E(π/2) = 2M for all a."""
        M = 1.0
        for a in [0.0, 0.3, 0.6, 0.9, 0.99]:
            r_E = ergosphere_radius(np.pi / 2, M, a)
            np.testing.assert_allclose(r_E, 2.0 * M, rtol=1e-14,
                                       err_msg=f"r_E(π/2) ≠ 2M for a={a}")

    def test_horizon_schwarzschild(self):
        """r_+ = 2M for a=0."""
        np.testing.assert_allclose(event_horizon(1.0, 0.0), 2.0, rtol=1e-15)

    def test_horizon_extremal(self):
        """r_+ = M for a = M (extremal)."""
        M = 1.0
        np.testing.assert_allclose(event_horizon(M, M), M, rtol=1e-14)

    @pytest.mark.parametrize("a, expected", [
        (0.0,   6.0),      # Schwarzschild: 6M
        (1.0,   1.0),      # Extremal prograde: M
        (1.0,   9.0),      # Extremal retrograde: 9M (tested separately)
    ])
    def test_isco_schwarzschild(self, a, expected):
        """ISCO radius at known limits."""
        if expected == 9.0:
            pytest.skip("Retrograde tested separately")
        np.testing.assert_allclose(isco_radius(1.0, a), expected, rtol=1e-10,
                                   err_msg=f"ISCO wrong for a={a}")

    def test_isco_retrograde_extremal(self):
        """Retrograde ISCO at a=M = 9M (Bardeen 1972)."""
        np.testing.assert_allclose(
            isco_radius(1.0, 1.0, prograde=False), 9.0, rtol=1e-10
        )

    def test_isco_monotonic_prograde(self):
        """Prograde ISCO decreases monotonically with spin."""
        a_arr = np.linspace(0.0, 0.99, 20)
        r_arr = np.array([isco_radius(1.0, a, prograde=True) for a in a_arr])
        assert np.all(np.diff(r_arr) < 0), "ISCO should decrease with spin"

    def test_is_in_ergosphere(self):
        """Point at ergosphere midpoint should be inside ergosphere."""
        M, a   = 1.0, 0.9
        r_plus = event_horizon(M, a)
        r_ergo = ergosphere_radius(np.pi / 2, M, a)
        r_mid  = 0.5 * (r_plus + r_ergo)
        assert is_in_ergosphere(r_mid, np.pi / 2, M, a)

    def test_not_in_ergosphere_outside(self):
        """Point far outside should not be in ergosphere."""
        assert not is_in_ergosphere(10.0, np.pi / 2, M=1.0, a=0.9)


# ---------------------------------------------------------------------------
# 7. Frame-dragging
# ---------------------------------------------------------------------------

class TestFrameDragging:
    def test_omega_positive_prograde(self):
        """ω > 0 everywhere for a > 0."""
        M, a = 1.0, 0.9
        r_X  = 0.5 * (event_horizon(M, a) + ergosphere_radius(np.pi / 2, M, a))
        omega = frame_drag_omega(r_X, np.pi / 2, M, a)
        assert omega > 0, f"ω = {omega} should be positive for a > 0"

    def test_omega_zero_schwarzschild(self):
        """ω = 0 for a = 0."""
        omega = frame_drag_omega(5.0, np.pi / 2, M=1.0, a=0.0)
        np.testing.assert_allclose(omega, 0.0, atol=1e-15)

    def test_lapse_positive(self):
        """Lapse function α > 0 outside the horizon."""
        M, a = 1.0, 0.9
        r = event_horizon(M, a) + 0.5
        alpha = lapse_function(r, np.pi / 2, M, a)
        assert alpha > 0, f"α = {alpha} should be > 0"

    def test_lapse_vanishes_at_horizon(self):
        """α → 0 as r → r_+ (Δ → 0)."""
        M, a  = 1.0, 0.9
        r_near = event_horizon(M, a) + 1e-6
        alpha  = lapse_function(r_near, np.pi / 2, M, a)
        assert alpha < 1e-3, f"α = {alpha} should be ≈ 0 near horizon"
