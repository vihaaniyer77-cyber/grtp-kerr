"""Analytical magnetic reconnection field geometries."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..spacetime.kerr import metric as kerr_metric, zamo_tetrad

__all__ = ["FieldProfile", "XPointField", "HarrisSheetField"]


# ---------------------------------------------------------------------------
# Utility: build F^(ab) from E and B components in the ZAMO frame
# ---------------------------------------------------------------------------

def _frame_tensor(
    E_r: float,   E_theta: float, E_phi: float,
    B_r: float,   B_theta: float, B_phi: float,
) -> NDArray:
    """
    Build the 4×4 antisymmetric EM tensor F^(ab) in the local ZAMO frame.

    Frame-index ordering: (t̂=0, r̂=1, θ̂=2, φ̂=3).

    Parameters
    ----------
    E_r, E_theta, E_phi : float   ZAMO-measured electric field components
    B_r, B_theta, B_phi : float   ZAMO-measured magnetic field components

    Returns
    -------
    F : (4, 4) ndarray,  antisymmetric
    """
    F = np.zeros((4, 4))
    # Electric field rows (t̂ ↔ spatial)
    F[0, 1] =  E_r;      F[1, 0] = -E_r
    F[0, 2] =  E_theta;  F[2, 0] = -E_theta
    F[0, 3] =  E_phi;    F[3, 0] = -E_phi
    # Magnetic field (spatial pairs)
    F[1, 2] =  B_phi;    F[2, 1] = -B_phi   #  F^r̂θ̂ = B^φ̂
    F[1, 3] = -B_theta;  F[3, 1] =  B_theta  #  F^r̂φ̂ = −B^θ̂
    F[2, 3] =  B_r;      F[3, 2] = -B_r      #  F^θ̂φ̂ = B^r̂
    return F


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class FieldProfile:
    """
    Abstract base class for EM field profiles in the Kerr ergosphere.

    Subclasses must implement `field_in_frame(r, theta)` returning F^(ab)
    in the ZAMO frame.  The BL-coordinate and mixed tensors are provided
    automatically via the ZAMO tetrad transform.
    """

    def __init__(self, M: float, a: float) -> None:
        self.M = float(M)
        self.a = float(a)

    # -- Interface to implement in subclasses --------------------------------

    def field_in_frame(self, r: float, theta: float) -> NDArray:
        """
        Return F^(ab) (4×4, antisymmetric) in the ZAMO orthonormal frame.
        Override in subclasses.
        """
        raise NotImplementedError

    # -- Derived quantities (do not override unless performance-critical) ----

    def em_tensor(self, r: float, theta: float) -> NDArray:
        """
        Contravariant EM tensor F^μν in Boyer-Lindquist coordinates.

        Computed via the ZAMO tetrad transform:
          F^μν = Σ_{a,b} e^μ_(a) e^ν_(b) F^(ab)_(frame)

        Parameters
        ----------
        r, theta : float   BL position

        Returns
        -------
        F_BL : (4, 4) ndarray,  antisymmetric
        """
        F_frame = self.field_in_frame(r, theta)     # (4,4) in ZAMO frame
        tet     = zamo_tetrad(r, theta, self.M, self.a)  # (4,4): tet[a, μ]
        # F^μν = e^μ_(a) e^ν_(b) F^(ab)
        return np.einsum("am,bn,ab->mn", tet, tet, F_frame)

    def em_tensor_mixed(self, r: float, theta: float) -> NDArray:
        """
        Mixed EM tensor F^μ_ν = F^μλ g_λν in BL coordinates.

        This is the form entering the Lorentz force:
          (q/m) F^μ_ν u^ν

        Returns
        -------
        F_mixed : (4, 4) ndarray
        """
        F_up  = self.em_tensor(r, theta)            # F^μν, shape (4,4)
        g_low = kerr_metric(r, theta, self.M, self.a)  # g_μν, shape (4,4)
        # F^μ_ν = F^μλ g_λν  ← lower the second index
        return np.einsum("ml,ln->mn", F_up, g_low)

    def em_invariants(self, r: float, theta: float) -> tuple[float, float]:
        """
        Lorentz-invariant scalars of the EM field:

          I₁ = ½ F_μν F^μν  (= B² − E² in the local frame)
          I₂ = ½ F_μν *F^μν = E·B (Lorentz scalar)

        Useful for characterising the reconnection site.
        """
        from ..spacetime.kerr import metric as gm, inverse_metric as gm_inv
        g    = gm(r, theta, self.M, self.a)
        ginv = gm_inv(r, theta, self.M, self.a)
        Fup  = self.em_tensor(r, theta)
        # Lower both indices: F_μν = g_μλ g_νσ F^λσ
        F_low = np.einsum("am,bn,mn->ab", g, g, Fup)
        I1 = 0.5 * np.einsum("ab,ab->", F_low, Fup)
        # Dual *F^μν = (1/2) ε^μναβ F_αβ  — skip for now; I2 computed from frame
        F_frame = self.field_in_frame(r, theta)
        # In ZAMO frame: E and B are direct frame components
        E_r, E_th, E_ph = F_frame[0, 1], F_frame[0, 2], F_frame[0, 3]
        B_r, B_th, B_ph = F_frame[2, 3], -F_frame[1, 3], F_frame[1, 2]
        I2 = E_r * B_r + E_th * B_th + E_ph * B_ph
        return float(I1), float(I2)


# ---------------------------------------------------------------------------
# 2D Magnetic X-point field  (PRIMARY)
# ---------------------------------------------------------------------------

class XPointField(FieldProfile):
    """
    2D magnetic null-point (X-point) reconnection field.

    Centred at (r_X, θ_X) in the ZAMO frame, with a Gaussian spatial
    envelope that confines the field to the current sheet.

    Field components in the ZAMO frame
    ------------------------------------
    Poloidal magnetic field (creates the 2D null point at the X-point):
      B^r̂  =  B₀ · (θ − θ_X) / λ · f(r, θ)
      B^θ̂  = −B₀ · (r − r_X) / λ · f(r, θ)

    Reconnecting electric field (drives particle acceleration along X-line):
      E^φ̂  =  E₀ · f(r, θ)

    Gaussian envelope:
      f(r, θ) = exp{−[(r − r_X)² + (r(θ − θ_X))²] / (2λ²)}

    Parameters
    ----------
    M, a    : float   Black hole mass and spin
    r_X     : float   X-point radial position (recommend inside ergosphere)
    theta_X : float   X-point polar angle [rad]  (default π/2, equatorial)
    B0      : float   Magnetic field strength amplitude  [geom. units]
    E0      : float   Reconnecting electric field amplitude
    lam     : float   Current sheet half-width  [r_g]

    Notes
    -----
    The magnetisation parameter σ = B²/(4πρ) is set implicitly through B0.
    For a typical high-magnetisation plasma (σ ≫ 1), set B0 ≫ 1 in
    geometrized units.  The ratio E0/B0 = ε is the reconnection rate;
    relativistic reconnection has ε ~ 0.1–0.5 (Lyubarsky 2005).
    """

    def __init__(
        self,
        M:       float,
        a:       float,
        r_X:     float,
        theta_X: float = np.pi / 2,
        B0:      float = 1.0,
        E0:      float = 0.1,
        lam:     float = 0.5,
    ) -> None:
        super().__init__(M, a)
        self.r_X     = float(r_X)
        self.theta_X = float(theta_X)
        self.B0      = float(B0)
        self.E0      = float(E0)
        self.lam     = float(lam)

    def _envelope(self, r: float, theta: float) -> float:
        """
        Gaussian localisation envelope f(r, θ).

        Arc-length in θ is approximated as r Δθ, consistent with the flat-space
        limit in a small patch around (r_X, θ_X).
        """
        dr = r - self.r_X
        ds = r * (theta - self.theta_X)   # arc-length offset in θ
        return np.exp(-(dr * dr + ds * ds) / (2.0 * self.lam ** 2))

    def field_in_frame(self, r: float, theta: float) -> NDArray:
        """
        F^(ab) in the local ZAMO frame at (r, θ).

        The field vanishes exponentially outside the current sheet
        (|r − r_X| ≫ λ  or  |θ − θ_X| ≫ λ/r).
        """
        f  = self._envelope(r, theta)
        dr = r - self.r_X
        dt = theta - self.theta_X

        B_r     = self.B0 * (dt / self.lam) * f
        B_theta = -self.B0 * (dr / self.lam) * f
        E_phi   =  self.E0 * f

        return _frame_tensor(
            E_r=0.0, E_theta=0.0, E_phi=E_phi,
            B_r=B_r, B_theta=B_theta, B_phi=0.0,
        )

    def reconnection_rate(self) -> float:
        """
        Reconnection rate (inflow-to-outflow ratio) at the X-point:
          ε = E₀ / B₀.
        Relativistic reconnection: ε ~ 0.1–0.5.
        """
        return self.E0 / self.B0 if self.B0 > 0 else float("inf")


# ---------------------------------------------------------------------------
# 1D Harris current sheet field  (ALTERNATE — comparison / validation)
# ---------------------------------------------------------------------------

class HarrisSheetField(FieldProfile):
    """
    1D Harris current-sheet profile.

    Used as an alternate field profile for:
      (a) Validating the Schwarzschild limit (a → 0) against published
          flat-spacetime Harris-sheet test-particle studies.
      (b) Providing a simpler baseline to isolate 2D X-point effects.

    Field components in the ZAMO frame
    ------------------------------------
    Reversing magnetic field (tanh profile across the sheet):
      B^θ̂  = B₀ · tanh[(r − r_X) / δ] · h(θ)

    Reconnecting electric field (maximum at the sheet centre):
      E^φ̂  = E₀ · sech²[(r − r_X) / δ] · h(θ)

    Angular Gaussian (localises sheet in θ):
      h(θ)  = exp{−[r_X (θ − θ_X)]² / (2λ²)}

    Parameters
    ----------
    delta : float   Sheet half-thickness in the radial direction [r_g].
                    Controls the steepness of the magnetic field reversal.
                    Typically δ < λ (thin sheet, large aspect ratio).

    All other parameters as for XPointField.
    """

    def __init__(
        self,
        M:       float,
        a:       float,
        r_X:     float,
        theta_X: float = np.pi / 2,
        B0:      float = 1.0,
        E0:      float = 0.1,
        lam:     float = 0.5,
        delta:   float = 0.2,
    ) -> None:
        super().__init__(M, a)
        self.r_X     = float(r_X)
        self.theta_X = float(theta_X)
        self.B0      = float(B0)
        self.E0      = float(E0)
        self.lam     = float(lam)
        self.delta   = float(delta)

    def field_in_frame(self, r: float, theta: float) -> NDArray:
        """F^(ab) in the local ZAMO frame (Harris-sheet profile)."""
        u       = (r - self.r_X) / self.delta
        tanh_u  = np.tanh(u)
        sech2_u = 1.0 / np.cosh(u) ** 2

        # Angular Gaussian envelope in θ
        dtheta  = self.r_X * (theta - self.theta_X)
        h_theta = np.exp(-dtheta * dtheta / (2.0 * self.lam ** 2))

        B_theta = self.B0 * tanh_u  * h_theta
        E_phi   = self.E0 * sech2_u * h_theta

        return _frame_tensor(
            E_r=0.0, E_theta=0.0, E_phi=E_phi,
            B_r=0.0, B_theta=B_theta, B_phi=0.0,
        )
