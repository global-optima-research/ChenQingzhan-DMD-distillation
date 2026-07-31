"""Shared style for final-report figures.

Project design system (approved deck semantics; overrides library defaults):
  NAVY      #1F3864  teacher / structure / primary series
  NAVY_SOFT #8FAADC  secondary series
  ORANGE    #C55A11  reserved: main degradation axis (diversity) + warnings;
                     at most one orange data highlight per figure
  TEAL      #00796B  GAN-off arm (avoid "orange = best" misreading)
  GRAY      #666666  annotations
Identity is never carried by color alone: line style / markers / direct
line-end labels back every distinction.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

NAVY = "#1F3864"
NAVY_SOFT = "#8FAADC"
ORANGE = "#C55A11"
TEAL = "#00796B"
GRAY = "#666666"
INK = "#222222"
GRID = "#DDDDDD"
NAVY_TINT = "#E8EDF6"   # light fill derived from NAVY for boxes
ORANGE_TINT = "#F5E0D3" # light fill derived from ORANGE for bands

FULL_W = 6.5  # single-column width (in)


def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 110,
        "savefig.dpi": 300,
    })


def save(fig, stem, outdir):
    """Save PDF (LaTeX) + PNG (300 dpi preview)."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{outdir}/{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"saved {stem}.pdf/.png")


def ygrid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def nonzero_note(ax, axis="y", loc=None):
    """Explicit '(non-zero origin)' marker for truncated axes."""
    if axis == "y":
        ax.annotate("(non-zero origin)", xy=(0, 0), xycoords="axes fraction",
                    xytext=(0.005, 0.015), textcoords="axes fraction",
                    fontsize=7, color=GRAY, ha="left", va="bottom")
    else:
        x, y = loc if loc else (0.995, -0.30)
        ax.annotate("(non-zero origin)", xy=(x, y), xycoords="axes fraction",
                    fontsize=7, color=GRAY, ha="right", va="top",
                    annotation_clip=False)


def axis_break(ax, x, y, height, color=INK):
    """Small double-slash glyph marking a truncated bar at axis start."""
    dx = 0.006 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    for off in (-0.8 * dx, 0.8 * dx):
        ax.plot([x + off - dx * 0.6, x + off + dx * 0.6],
                [y - height * 0.55, y + height * 0.55],
                color="white", lw=2.2, zorder=6, clip_on=False)
        ax.plot([x + off - dx * 0.6, x + off + dx * 0.6],
                [y - height * 0.55, y + height * 0.55],
                color=color, lw=0.7, zorder=7, clip_on=False)
