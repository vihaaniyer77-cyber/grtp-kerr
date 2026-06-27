"""Plotting routines for 2D basin-of-fate maps."""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from numpy.typing import NDArray

from ..analysis.topology import (
    ESCAPE_POSITIVE, ESCAPE_NEGATIVE, PLUNGE, TRAPPED, fate_color
)
from .trajectory import apply_mnras_style

__all__ = ["plot_basin_map", "fate_grid_to_rgba"]


# Ordered list of fate labels and their colors
_FATE_ORDER = [ESCAPE_POSITIVE, ESCAPE_NEGATIVE, PLUNGE, TRAPPED, "failed"]
_FATE_INT   = {f: i for i, f in enumerate(_FATE_ORDER)}
_RGBA       = np.array([mcolors.to_rgba(fate_color(f)) for f in _FATE_ORDER])


def fate_grid_to_rgba(fate_grid: NDArray) -> NDArray:
    """
    Convert a 2D grid of fate strings to an RGBA image array.

    Parameters
    ----------
    fate_grid : (N_r, N_ur) ndarray of str   Grid of fate labels

    Returns
    -------
    rgba : (N_r, N_ur, 4) ndarray   RGBA float image in [0,1]
    """
    N_r, N_ur = fate_grid.shape
    idx  = np.vectorize(lambda f: _FATE_INT.get(f, _FATE_INT["failed"]))(fate_grid)
    rgba = _RGBA[idx]   # shape (N_r, N_ur, 4)
    return rgba.astype(float)


def plot_basin_map(
    fate_grid:    NDArray,
    r_values:     NDArray,
    ur_values:    NDArray,
    M:            float,
    a:            float,
    r_X:          float,
    ax:           Optional[plt.Axes] = None,
    title:        Optional[str] = None,
    figsize:      tuple = (8, 7),
    show_xpoint:  bool = True,
    show_horizon: bool = True,
    interpolation: str = "none",
) -> plt.Figure:
    """
    Render the basin-of-fate map as a colour-coded 2D grid.

    Parameters
    ----------
    fate_grid  : (N_r, N_ur) ndarray of str
        Grid of fate labels in (r, u^r) phase space.
        Axis 0 = r (rows, vertical axis), Axis 1 = u^r (cols, horizontal).
    r_values   : (N_r,) ndarray   Radial coordinate grid
    ur_values  : (N_ur,) ndarray  u^r coordinate grid
    M, a       : float            Black hole parameters
    r_X        : float            X-point radius (marked on plot)
    title      : str, optional    Default auto-generated
    show_xpoint : bool   Mark the X-point location with a white cross
    show_horizon: bool   Draw a horizontal line at r = r_+
    interpolation : str  imshow interpolation method

    Returns
    -------
    fig : plt.Figure
    """
    apply_mnras_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0D0D0D")
    else:
        fig = ax.figure

    ax.set_facecolor("#0D0D0D")

    # Convert fate strings → RGBA and display
    rgba = fate_grid_to_rgba(fate_grid)

    # extent: [ur_min, ur_max, r_min, r_max] for imshow with origin='lower'
    extent = [
        ur_values[0], ur_values[-1],
        r_values[0],  r_values[-1],
    ]
    ax.imshow(
        rgba,
        origin        = "lower",
        extent        = extent,
        aspect        = "auto",
        interpolation = interpolation,
        zorder        = 1,
    )

    # --- Overlays ---
    from ..spacetime.ergosphere import event_horizon
    r_plus = event_horizon(M, a)

    if show_horizon:
        ax.axhline(r_plus, color="white", lw=1.0, ls="--", alpha=0.8,
                   label=r"$r_+$", zorder=4)
        ax.axhline(2.0 * M, color="#FFD700", lw=0.8, ls=":", alpha=0.6,
                   label=r"$r_E(\pi/2)$", zorder=4)

    if show_xpoint:
        ax.axhline(r_X, color="#FFFFFF", lw=0.6, ls="-.", alpha=0.4,
                   label=r"$r_X$ (X-point)", zorder=4)
        ax.scatter(
            [0.0], [r_X],
            marker="x", s=80, c="white", zorder=5,
            label="X-point", linewidths=1.5,
        )

    # --- Legend ---
    legend_patches = [
        Patch(facecolor=fate_color(ESCAPE_POSITIVE), label="Escape (ΔE > 0)"),
        Patch(facecolor=fate_color(ESCAPE_NEGATIVE), label="Escape (ΔE ≤ 0)"),
        Patch(facecolor=fate_color(PLUNGE),          label="Plunge"),
        Patch(facecolor=fate_color(TRAPPED),         label="Trapped"),
    ]
    ax.legend(
        handles    = legend_patches,
        loc        = "upper right",
        framealpha = 0.25,
        labelcolor = "white",
        facecolor  = "#0D0D0D",
        edgecolor  = "#444444",
        fontsize   = 8,
    )

    # --- Axis labels and title ---
    if title is None:
        title = rf"Basin of Fate  ($a/M = {a:.2f}$,  $r_X = {r_X:.2f}\,M$)"
    ax.set_xlabel(r"Initial radial 4-velocity $u^r_0$", color="white")
    ax.set_ylabel(r"Initial radius $r_0 \; [M]$",        color="white")
    ax.set_title(title, color="white", pad=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    fig.tight_layout()
    return fig


def plot_efficiency_heatmap(
    efficiency_grid:  NDArray,
    a_values:         NDArray,
    rX_values:        NDArray,
    ax:               Optional[plt.Axes] = None,
    title:            str = r"Penrose Efficiency $\eta(a, r_X)$",
    figsize:          tuple = (8, 6),
    vmin:             float = 0.0,
    vmax:             float = 1.0,
    cbar_label:       str = r"Efficiency $\eta$",
) -> plt.Figure:
    """
    Heatmap of the Penrose efficiency η = N_escape_positive / N_total
    as a function of (a/M, r_X/M).

    Parameters
    ----------
    efficiency_grid : (N_a, N_rX) ndarray   η values in [0, 1]
    a_values        : (N_a,) ndarray        Spin parameter values
    rX_values       : (N_rX,) ndarray       X-point radii

    Notes
    -----
    The X-point grid is already constrained to r_+ < r_X < r_erg(π/2) = 2M
    by the fractional parameterisation in exp1_efficiency_sweep.py.  We draw
    the r_erg = 2M boundary line as an explicit visual reference to confirm
    this domain restriction.  Any non-zero η appearing to the right of this
    line would be a physical error — none should be present.
    """
    apply_mnras_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor="#0D0D0D")
    else:
        fig = ax.figure
    ax.set_facecolor("#0D0D0D")

    # Use plasma colormap: continuous gradient reveals fine structure in η.
    # The previous binary-looking render with 'inferno' at a narrow dynamic
    # range was masking the gradient — plasma + explicit vmax fixes this.
    if rX_values.ndim == 1:
        # Simple 1D r_X grid: pcolormesh works cleanly.
        rX_mesh, a_mesh = np.meshgrid(rX_values, a_values)
        im = ax.pcolormesh(
            rX_mesh, a_mesh, efficiency_grid,
            cmap="plasma", vmin=vmin, vmax=vmax, shading="auto", zorder=1,
        )
    else:
        # 2D rX_grid: each spin row has different absolute r_X values.
        # pcolormesh raises a monotonicity warning on 2D non-uniform grids.
        # Instead, use imshow with extent=[rX_min, rX_max, a_min, a_max]
        # derived from the column-wise mean r_X per fractional position.
        rX_col_mean = np.nanmean(rX_values, axis=0)   # shape (N_rX,)
        extent = [
            float(rX_col_mean[0]),  float(rX_col_mean[-1]),
            float(a_values[0]),     float(a_values[-1]),
        ]
        im = ax.imshow(
            efficiency_grid,
            origin="lower", extent=extent, aspect="auto",
            cmap="plasma", vmin=vmin, vmax=vmax, zorder=1,
        )
        # Correct x-axis to show actual (mean) rX column values
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel(r"X-point radius $r_X \; [M]$",   color="white")
    ax.set_ylabel(r"Black hole spin $a/M$",           color="white")
    ax.set_title(title, color="white", pad=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # Ergosphere equatorial boundary: r_erg(θ=π/2) = 1 + √(1 - a²cos²θ)|_{θ=π/2} = 2M.
    # All valid X-points must lie to the LEFT of this line.
    # The vertical line acts as an explicit domain-boundary guard for the reader.
    ax.axvline(2.0, color="#FFD700", lw=1.2, ls="--", alpha=0.85, zorder=3,
               label=r"Ergosphere boundary $r_{\rm erg}(\pi/2) = 2M$")
    ax.legend(loc="upper left", framealpha=0.25, labelcolor="white",
              facecolor="#0D0D0D", edgecolor="#444444", fontsize=8)

    # Contour lines at physically meaningful efficiency levels (1%, 2%, 3%, 5%).
    # These replace the old unlabeled 10%/20%/30% contours which were above the
    # actual data range and therefore never appeared.
    if efficiency_grid.size > 4:
        try:
            _eta_max = float(np.nanmax(efficiency_grid))
            _levels = [lv for lv in [0.01, 0.02, 0.03, 0.05] if lv < _eta_max]
            if _levels:
                cs = ax.contour(rX_mesh, a_mesh, efficiency_grid,
                                levels=_levels,
                                colors="white", alpha=0.5, linewidths=0.8)
                ax.clabel(cs, fmt=lambda v: f"{v*100:.0f}%",
                          colors="white", fontsize=7)
        except Exception:
            pass

    fig.tight_layout()
    return fig
