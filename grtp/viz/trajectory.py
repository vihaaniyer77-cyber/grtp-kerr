"""Plotting routines for 2D and 3D particle trajectories."""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from numpy.typing import NDArray

from ..spacetime.ergosphere import event_horizon, ergosphere_radius

__all__ = [
    "plot_trajectory_2d",
    "plot_trajectory_3d",
    "plot_energy_evolution",
    "apply_mnras_style",
]

# ---------------------------------------------------------------------------
# Global style helpers
# ---------------------------------------------------------------------------

def apply_mnras_style() -> None:
    """
    Apply MNRAS-compatible matplotlib rcParams.

    Font sizes and line widths are calibrated for a two-column MNRAS figure
    (width ≈ 8.5 cm per column).  Figures should be saved as PDF or EPS.
    """
    mpl.rcParams.update({
        "font.family"       : "serif",
        "font.size"         : 9,
        "axes.titlesize"    : 10,
        "axes.labelsize"    : 9,
        "xtick.labelsize"   : 8,
        "ytick.labelsize"   : 8,
        "legend.fontsize"   : 8,
        "lines.linewidth"   : 1.2,
        "axes.linewidth"    : 0.8,
        "xtick.major.width" : 0.8,
        "ytick.major.width" : 0.8,
        "figure.dpi"        : 150,
        "savefig.dpi"       : 300,
        "savefig.bbox"      : "tight",
        "text.usetex"       : False,   # set True if LaTeX available
    })

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _bl_to_cartesian(x_arr: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Convert Boyer-Lindquist (r, θ, φ) to Cartesian (X, Y, Z)."""
    r     = x_arr[:, 1]
    theta = x_arr[:, 2]
    phi   = x_arr[:, 3]
    X = r * np.sin(theta) * np.cos(phi)
    Y = r * np.sin(theta) * np.sin(phi)
    Z = r * np.cos(theta)
    return X, Y, Z


def _ergosphere_contour(M: float, a: float, n: int = 360) -> tuple[NDArray, NDArray]:
    """(X, Z) contour of the equatorial ergosphere cross-section."""
    theta_arr = np.linspace(0.0, np.pi, n)
    r_E  = np.array([ergosphere_radius(th, M, a) for th in theta_arr])
    X_E  = r_E * np.sin(theta_arr)
    Z_E  = r_E * np.cos(theta_arr)
    return X_E, Z_E


# ---------------------------------------------------------------------------
# 2D trajectory plot (meridional plane projection)
# ---------------------------------------------------------------------------

def plot_trajectory_2d(
    trajectories: Union[list, object],
    M:             float,
    a:             float,
    ax:            Optional[plt.Axes] = None,
    title:         str = "Particle Trajectories — Meridional Plane",
    fate_colors:   Optional[dict] = None,
    show_grid:     bool = True,
    figsize:       tuple = (7, 7),
) -> plt.Figure:
    """
    Plot particle trajectories projected onto the meridional plane (r sin θ, r cos θ).

    The ergosphere and event horizon are overlaid.  Trajectories are colour-coded
    by fate (escape=blue, plunge=red, trapped=gold) if fate information is
    available on the trajectory objects.

    Parameters
    ----------
    trajectories : Trajectory or list[Trajectory]
    M, a         : float   Black hole mass and spin
    ax           : plt.Axes, optional   Re-use an existing axes
    title        : str
    fate_colors  : dict, optional   Override default fate-to-color mapping
    figsize      : tuple

    Returns
    -------
    fig : plt.Figure
    """
    apply_mnras_style()

    if not isinstance(trajectories, list):
        trajectories = [trajectories]

    _default_colors = {
        "escape":          "#2E86AB",
        "escape_positive": "#2E86AB",
        "escape_negative": "#89CFF0",
        "plunge":          "#C0392B",
        "trapped":         "#D4AC0D",
    }
    if fate_colors:
        _default_colors.update(fate_colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0D0D0D")
    else:
        fig = ax.figure

    ax.set_facecolor("#0D0D0D")

    # --- Ergosphere (filled region) ---
    theta_arr = np.linspace(0.0, np.pi, 360)
    r_E  = np.array([ergosphere_radius(th, M, a) for th in theta_arr])
    r_H  = event_horizon(M, a)

    Xe   = r_E * np.sin(theta_arr)
    Ze   = r_E * np.cos(theta_arr)
    Xh   = r_H * np.sin(theta_arr)
    Zh   = r_H * np.cos(theta_arr)

    # Mirror for both hemispheres (ϕ = 0 side)
    ax.fill(np.concatenate([Xe, Xe[::-1]]),
            np.concatenate([Ze, -Ze[::-1]]),
            alpha=0.15, color="#FFD700", label="Ergosphere", zorder=1)
    ax.fill(np.concatenate([Xh, Xh[::-1]]),
            np.concatenate([Zh, -Zh[::-1]]),
            alpha=0.8, color="#1A1A2E", label="Event horizon", zorder=2)
    ax.plot(np.concatenate([Xe, Xe[::-1]]),
            np.concatenate([Ze, -Ze[::-1]]),
            color="#FFD700", lw=0.8, alpha=0.6, zorder=3)
    ax.plot(np.concatenate([Xh, Xh[::-1]]),
            np.concatenate([Zh, -Zh[::-1]]),
            color="#AAAAAA", lw=0.8, zorder=3)

    # --- Trajectories ---
    for traj in trajectories:
        fate  = getattr(traj, "fate", "trapped")
        color = _default_colors.get(fate, "#888888")
        rho   = traj.r * np.sin(traj.theta)     # cylindrical radius
        z_bl  = traj.r * np.cos(traj.theta)

        ax.plot(rho, z_bl, color=color, lw=0.8, alpha=0.85, zorder=4)
        # Mark starting point
        ax.scatter(rho[0], z_bl[0], color=color, s=20, zorder=5,
                   edgecolors="white", linewidths=0.5)

    # --- Formatting ---
    r_max = max(t.r.max() for t in trajectories) * 1.1
    r_max = min(r_max, 12.0 * M)
    ax.set_xlim(-0.1, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\rho = r\sin\theta \; [M]$", color="white")
    ax.set_ylabel(r"$z = r\cos\theta \; [M]$",    color="white")
    ax.set_title(title, color="white", pad=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    if show_grid:
        ax.grid(color="#333333", lw=0.4, alpha=0.5)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2E86AB", lw=1.5, label="Escape (ΔE > 0)"),
        Line2D([0], [0], color="#C0392B", lw=1.5, label="Plunge"),
        Line2D([0], [0], color="#D4AC0D", lw=1.5, label="Trapped"),
        patches.Patch(facecolor="#FFD700", alpha=0.3, label="Ergosphere"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              framealpha=0.2, labelcolor="white",
              facecolor="#1A1A2E", edgecolor="#444444")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3D trajectory plot (Cartesian embedding)
# ---------------------------------------------------------------------------

def plot_trajectory_3d(
    trajectories: Union[list, object],
    M:             float,
    a:             float,
    title:         str = "3D Trajectory in Kerr Spacetime",
    figsize:       tuple = (9, 8),
    elev:          float = 25.0,
    azim:          float = -60.0,
) -> plt.Figure:
    """
    3D Cartesian trajectory plot with ergosphere wireframe.

    Boyer-Lindquist (r, θ, φ) are embedded in Cartesian (X, Y, Z) via
    the standard transformation X = r sinθ cosφ, etc.

    Parameters
    ----------
    elev, azim : float   Initial viewing angles for mpl_toolkits Axes3D
    """
    apply_mnras_style()
    from mpl_toolkits.mplot3d import Axes3D

    if not isinstance(trajectories, list):
        trajectories = [trajectories]

    fig = plt.figure(figsize=figsize, facecolor="#0D0D0D")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0D0D0D")
    ax.view_init(elev=elev, azim=azim)

    # --- Ergosphere surface wireframe ---
    u_surf = np.linspace(0, np.pi, 40)      # θ
    v_surf = np.linspace(0, 2 * np.pi, 60)  # φ
    uu, vv = np.meshgrid(u_surf, v_surf)
    R_E = np.vectorize(lambda th: ergosphere_radius(th, M, a))(uu)
    XE  = R_E * np.sin(uu) * np.cos(vv)
    YE  = R_E * np.sin(uu) * np.sin(vv)
    ZE  = R_E * np.cos(uu)
    ax.plot_wireframe(XE, YE, ZE, color="#FFD700", alpha=0.08,
                      linewidth=0.3, zorder=1)

    # --- Event horizon surface ---
    R_H = event_horizon(M, a)
    XH  = R_H * np.sin(uu) * np.cos(vv)
    YH  = R_H * np.sin(uu) * np.sin(vv)
    ZH  = R_H * np.cos(uu)
    ax.plot_surface(XH, YH, ZH, color="#111111", alpha=0.9,
                    linewidth=0, zorder=2)

    # --- Trajectories ---
    _colors = {
        "escape":          "#2E86AB",
        "escape_positive": "#2E86AB",
        "escape_negative": "#89CFF0",
        "plunge":          "#C0392B",
        "trapped":         "#D4AC0D",
    }
    for traj in trajectories:
        fate       = getattr(traj, "fate", "trapped")
        color      = _colors.get(fate, "#AAAAAA")
        X, Y, Z    = _bl_to_cartesian(traj.x)
        ax.plot(X, Y, Z, color=color, lw=0.9, alpha=0.85, zorder=5)
        ax.scatter([X[0]], [Y[0]], [Z[0]], color=color, s=30, zorder=6,
                   edgecolors="white", linewidths=0.5)

    # --- Formatting ---
    for ax_part in [ax.xaxis, ax.yaxis, ax.zaxis]:
        ax_part.pane.fill  = False
        ax_part.pane.set_edgecolor("#333333")
        ax_part.label.set_color("white")
        ax_part._axinfo["grid"]["color"] = "#333333"
        ax_part._axinfo["grid"]["linewidth"] = 0.3

    ax.set_xlabel("X [M]", labelpad=6, color="white")
    ax.set_ylabel("Y [M]", labelpad=6, color="white")
    ax.set_zlabel("Z [M]", labelpad=6, color="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.set_title(title, color="white", pad=12)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Energy evolution plot
# ---------------------------------------------------------------------------

def plot_energy_evolution(
    trajectories: Union[list, object],
    M:             float,
    a:             float,
    ax:            Optional[plt.Axes] = None,
    title:         str = "Killing Energy E_∞ vs Proper Time",
    figsize:       tuple = (8, 4),
) -> plt.Figure:
    """
    Plot E_∞(τ) along one or more trajectories.

    Negative-energy regions are shaded red; the ergosphere crossing times are
    indicated.  The horizontal dashed line at E_∞ = 0 separates bound and
    unbound states.

    Parameters
    ----------
    trajectories : Trajectory or list[Trajectory]
    """
    apply_mnras_style()

    if not isinstance(trajectories, list):
        trajectories = [trajectories]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0D0D0D")
    else:
        fig = ax.figure

    ax.set_facecolor("#0D0D0D")

    _colors = {
        "escape":          "#2E86AB",
        "escape_positive": "#2E86AB",
        "escape_negative": "#89CFF0",
        "plunge":          "#C0392B",
        "trapped":         "#D4AC0D",
    }

    for traj in trajectories:
        fate  = getattr(traj, "fate", "trapped")
        color = _colors.get(fate, "#AAAAAA")
        ax.plot(traj.tau, traj.E_inf, color=color, lw=1.0, alpha=0.9)
        # Shade negative-energy region
        ax.fill_between(traj.tau, traj.E_inf, 0.0,
                        where=(traj.E_inf < 0),
                        color="#C0392B", alpha=0.15)

    # E_∞ = 0 reference line
    ax.axhline(0.0, color="#AAAAAA", lw=0.8, ls="--", alpha=0.7,
               label=r"$E_\infty = 0$")

    ax.set_xlabel(r"Proper time $\tau \; [M]$", color="white")
    ax.set_ylabel(r"Killing energy $E_\infty$",  color="white")
    ax.set_title(title, color="white", pad=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.grid(color="#333333", lw=0.4, alpha=0.5)
    ax.legend(loc="best", framealpha=0.2, labelcolor="white",
              facecolor="#1A1A2E", edgecolor="#444444")

    fig.tight_layout()
    return fig
