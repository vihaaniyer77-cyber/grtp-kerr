"""
grtp/analysis/topology.py
=========================
Orbit topology classification and basin-of-fate statistics.

Every test-particle trajectory ends in one of three fates:

  ESCAPE  — r exceeds r_escape: particle reaches (effective) infinity with
             kinetic energy, potentially carrying away net energy from the BH.
  PLUNGE  — r drops below r_+ + buffer: particle crosses the event horizon.
  TRAPPED — Neither escape nor plunge within τ_max: orbit is in a long-lived
             resonance or is a quasi-periodic orbit near the reconnection layer.

The ESCAPE class is further divided into two astrophysically distinct outcomes:
  ESCAPE_POSITIVE: E_∞_final > E_∞_initial  (net energy gain — Penrose process)
  ESCAPE_NEGATIVE: E_∞_final < E_∞_initial  (particle carries less energy away)

The basin-of-fate map (Experiment 2) plots these outcomes as a function of
initial conditions (r_0, u^r_0) in the phase space near the X-point.
Fractal basin boundaries are a signature of chaotic dynamics and are directly
related to the positive Lyapunov exponent measured in Experiment 3.

References
----------
  Kopáček & Karas (2014), ApJ 787, 117  — basin of attraction maps in Kerr
  Contopoulos & Harsoula (2010), IJBC 20, 2005  — fractal basin boundaries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ESCAPE", "PLUNGE", "TRAPPED",
    "ESCAPE_POSITIVE", "ESCAPE_NEGATIVE",
    "FateClassifier", "FateStats", "classify_population",
]

# Fate string constants
ESCAPE          = "escape"
PLUNGE          = "plunge"
TRAPPED         = "trapped"
ESCAPE_POSITIVE = "escape_positive"
ESCAPE_NEGATIVE = "escape_negative"

_FATE_INT = {
    ESCAPE_POSITIVE: 0,
    ESCAPE_NEGATIVE: 1,
    PLUNGE:          2,
    TRAPPED:         3,
    "failed":        4,
}



# ---------------------------------------------------------------------------
# Per-trajectory fate classifier
# ---------------------------------------------------------------------------

class FateClassifier:
    """
    Classify the fate and energy outcome of a single trajectory.

    The raw `fate` string from `Trajectory` (escape/plunge/trapped) is
    supplemented with the energy gain sub-classification.

    Parameters
    ----------
    energy_gain_threshold : float
        Minimum fractional energy gain ΔE/|E₀| to declare ESCAPE_POSITIVE.
        Default 0.01 (1% gain — above numerical noise).
    """

    def __init__(self, energy_gain_threshold: float = 0.01) -> None:
        self._eta_thr = energy_gain_threshold

    def classify(self, trajectory) -> str:
        """
        Return the detailed fate string for a Trajectory.

        Returns one of:
          'escape_positive', 'escape_negative', 'plunge', 'trapped'
        """
        if trajectory.fate == ESCAPE:
            if trajectory.energy_gain >= self._eta_thr:
                return ESCAPE_POSITIVE
            else:
                return ESCAPE_NEGATIVE
        return trajectory.fate   # 'plunge' or 'trapped'

    def energy_gain(self, trajectory) -> float:
        """Fractional Killing energy change ΔE_∞ / |E_∞(τ=0)|."""
        return trajectory.energy_gain

    def final_energy(self, trajectory) -> float:
        """Killing energy at the final trajectory point."""
        return float(trajectory.E_inf[-1])

    def is_negative_energy(self, trajectory) -> bool:
        """True if any point in the trajectory has E_∞ < 0."""
        return bool(np.any(trajectory.E_inf < 0.0))

    def min_energy(self, trajectory) -> float:
        """Minimum Killing energy attained along the trajectory."""
        return float(trajectory.E_inf.min())


# ---------------------------------------------------------------------------
# Population-level statistics
# ---------------------------------------------------------------------------

@dataclass
class FateStats:
    """
    Summary statistics for a population of trajectories.

    Attributes
    ----------
    n_total          : int    Total number of trajectories
    n_escape         : int    Total escape count (positive + negative)
    n_escape_positive: int    Escape with net energy gain
    n_escape_negative: int    Escape with net energy loss
    n_plunge         : int    Plunge into horizon
    n_trapped        : int    Trapped within τ_max
    efficiency       : float  η = n_escape_positive / n_total  [Penrose efficiency]
    mean_energy_gain : float  Mean ΔE/|E₀| over escape_positive trajectories
    mean_min_energy  : float  Mean minimum E_∞ (probes negative-energy excursions)
    fates            : list   Detailed fate string per trajectory
    energy_gains     : ndarray  ΔE/|E₀| per trajectory
    """
    n_total:           int
    n_escape:          int
    n_escape_positive: int
    n_escape_negative: int
    n_plunge:          int
    n_trapped:         int
    efficiency:        float
    mean_energy_gain:  float
    mean_min_energy:   float
    fates:             list  = field(default_factory=list)
    energy_gains:      Optional[NDArray] = None

    @classmethod
    def from_trajectories(
        cls,
        trajectories: list,
        classifier: Optional[FateClassifier] = None,
    ) -> "FateStats":
        """
        Compute FateStats from a list of Trajectory objects.

        Parameters
        ----------
        trajectories : list[Trajectory]   (None entries are skipped)
        classifier   : FateClassifier, optional
        """
        if classifier is None:
            classifier = FateClassifier()

        fates        = []
        gains        = []
        min_energies = []

        for traj in trajectories:
            if traj is None:
                fates.append("failed")
                gains.append(float("nan"))
                min_energies.append(float("nan"))
                continue
            fates.append(classifier.classify(traj))
            gains.append(classifier.energy_gain(traj))
            min_energies.append(classifier.min_energy(traj))

        fates_arr = np.array(fates)
        gains_arr = np.array(gains)

        n_total           = len(fates)
        n_escape_positive = int(np.sum(fates_arr == ESCAPE_POSITIVE))
        n_escape_negative = int(np.sum(fates_arr == ESCAPE_NEGATIVE))
        n_escape          = n_escape_positive + n_escape_negative
        n_plunge          = int(np.sum(fates_arr == PLUNGE))
        n_trapped         = int(np.sum(fates_arr == TRAPPED))
        efficiency        = n_escape_positive / max(n_total, 1)

        pos_mask = fates_arr == ESCAPE_POSITIVE
        mean_gain = float(np.nanmean(gains_arr[pos_mask])) if pos_mask.any() else 0.0
        mean_min  = float(np.nanmean(min_energies))

        return cls(
            n_total           = n_total,
            n_escape          = n_escape,
            n_escape_positive = n_escape_positive,
            n_escape_negative = n_escape_negative,
            n_plunge          = n_plunge,
            n_trapped         = n_trapped,
            efficiency        = efficiency,
            mean_energy_gain  = mean_gain,
            mean_min_energy   = mean_min,
            fates             = list(fates),
            energy_gains      = gains_arr,
        )

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        return (
            f"Population fate summary  (N = {self.n_total})\n"
            f"  Escape (positive ΔE): {self.n_escape_positive:5d}  "
            f"({100*self.n_escape_positive/max(self.n_total,1):.1f}%)\n"
            f"  Escape (negative ΔE): {self.n_escape_negative:5d}  "
            f"({100*self.n_escape_negative/max(self.n_total,1):.1f}%)\n"
            f"  Plunge             : {self.n_plunge:5d}  "
            f"({100*self.n_plunge/max(self.n_total,1):.1f}%)\n"
            f"  Trapped            : {self.n_trapped:5d}  "
            f"({100*self.n_trapped/max(self.n_total,1):.1f}%)\n"
            f"  Penrose efficiency η: {self.efficiency:.4f}\n"
            f"  Mean ΔE/E₀ (esc+)  : {self.mean_energy_gain:+.4f}\n"
            f"  Mean min(E_∞)      : {self.mean_min_energy:.4f}"
        )


# ---------------------------------------------------------------------------
# Convenience wrapper for batch classification
# ---------------------------------------------------------------------------

def classify_population(
    trajectories:  list,
    classifier:    Optional[FateClassifier] = None,
) -> FateStats:
    """
    Classify a list of trajectories and return summary statistics.

    Parameters
    ----------
    trajectories : list[Trajectory]   Output of `integrate_batch`
    classifier   : FateClassifier, optional

    Returns
    -------
    stats : FateStats
    """
    return FateStats.from_trajectories(trajectories, classifier)


def fate_color(fate: str) -> str:
    """
    Matplotlib color string for a given fate label.

    Used consistently across all basin-of-fate plots:
      escape_positive → steelblue
      escape_negative → skyblue
      plunge          → firebrick
      trapped         → goldenrod
      failed          → lightgrey
    """
    _colors = {
        ESCAPE_POSITIVE: "#2E86AB",   # steel blue
        ESCAPE_NEGATIVE: "#89CFF0",   # baby blue
        PLUNGE:          "#C0392B",   # deep red
        TRAPPED:         "#D4AC0D",   # golden
        "failed":        "#CCCCCC",   # grey
    }
    return _colors.get(fate, "#888888")
