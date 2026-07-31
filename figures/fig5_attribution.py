"""F5: attribution of the diversity collapse — three factors ruled out,
the collapse survives every ablation.

Every plotted number is transcribed verbatim from frozen local sources:
  teacher diversity 0.732, E1a 0.635, E1b 0.628, W1 0.649, W7 0.598/0.613:
      experiments/results/2026-07-20-g2-relay-vs-direct-final.md, table lines 12-17
  three-arm (relay-lineage) diversity interval 0.586-0.613 and E2b@500 0.590:
      experiments/results/2026-07-25-e2b-fulltable-ch3-threearm.md, lines 9 and 20
      ("三臂全部 0.586-0.613,均低于直蒸 E1a 的 0.635")
  band = min-max over all student best-of-sweep values across both files:
      [0.586 (three-arm interval low, = E2a@2000), 0.649 (W1@1000)]
Ruled-out statements: 07-20 fact ruling 2 (both direct arms also collapse);
07-25 ruling 4 (diversity insensitive to GAN on/off and to (t,eps) pairing).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import (NAVY, ORANGE, GRAY, INK, ORANGE_TINT,
                      FULL_W, apply_style, save, nonzero_note, axis_break)

OUT = "/Users/x3y/Desktop/ChenQingzhan-DMD-distillation/figures"

TEACHER_DIV = 0.732                                   # 07-20 L12
# member ticks: E2a interval low, E2b@500, W7@500, W7@1000, E1b@500, E1a@1000, W1@1000
STUDENT_DIV = [0.586, 0.590, 0.598, 0.613, 0.628, 0.635, 0.649]

BOXES = [
    ("Step-count relay (50→8→4)", "relay vs. direct: collapse in both"),
    ("GAN discriminator branch", "on vs. off: collapse unchanged"),
    ("Shared (t,ε) pairing", "shared vs. independent: unchanged"),
]

ROW_T, ROW_S = 1.0, 0.0
BAR_H = 0.42


def left_panel(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 106)
    ax.axis("off")
    centers = [84, 50, 16]
    for (title, sub), yc in zip(BOXES, centers):
        ax.add_patch(FancyBboxPatch(
            (2, yc - 13), 82, 26, boxstyle="round,pad=0,rounding_size=2.5",
            facecolor="white", edgecolor=NAVY, linewidth=0.9, zorder=3))
        ax.text(6, yc + 4.5, title, fontsize=7.5, color=INK, ha="left",
                va="center", fontweight="bold", zorder=4)
        ax.text(6, yc - 5.5, sub, fontsize=7, color=GRAY, ha="left",
                va="center", zorder=4)
        # "ruled out" stamp chip straddling the box's top-right corner
        ax.add_patch(FancyBboxPatch(
            (64.5, yc + 8), 21.5, 10, boxstyle="round,pad=0,rounding_size=2",
            facecolor="white", edgecolor=GRAY, linewidth=0.8, zorder=5,
            clip_on=False))
        ax.text(75.25, yc + 13, "ruled out", fontsize=7, color=GRAY,
                ha="center", va="center", style="italic", zorder=6,
                clip_on=False)
        # converging arrow toward the surviving-effect panel
        ax.add_patch(FancyArrowPatch(
            (85, yc), (99.5, 50), connectionstyle="arc3,rad={:+.2f}".format(
                0.0 if yc == 50 else (-0.22 if yc > 50 else 0.22)),
            arrowstyle="-|>", mutation_scale=8, color=GRAY, lw=0.9,
            shrinkA=2, shrinkB=0, zorder=2))


def right_panel(ax):
    XLO, XHI = 0.56, 0.755
    ax.barh(ROW_T, TEACHER_DIV - XLO, left=XLO, height=BAR_H, color=NAVY,
            zorder=3)
    ax.plot([TEACHER_DIV], [ROW_T + 0.42], marker="v", ms=5, color=NAVY,
            zorder=5, clip_on=False)
    ax.text(TEACHER_DIV - 0.008, ROW_T + 0.45, "teacher 0.732", fontsize=7,
            color=INK, ha="right", va="center")
    axis_break(ax, XLO + 0.007, ROW_T, BAR_H)
    # ORANGE band: all students, every ablation arm included
    ax.barh(ROW_S, max(STUDENT_DIV) - min(STUDENT_DIV), left=min(STUDENT_DIV),
            height=BAR_H, color=ORANGE_TINT, edgecolor=ORANGE, linewidth=0.7,
            zorder=3)
    for m in STUDENT_DIV:
        ax.plot([m, m], [ROW_S - 0.30 * BAR_H, ROW_S + 0.30 * BAR_H],
                color=ORANGE, lw=0.9, zorder=4)
    ax.text(min(STUDENT_DIV), ROW_S - 0.52, "0.586", fontsize=7, color=INK,
            ha="center", va="top")
    ax.text(max(STUDENT_DIV), ROW_S - 0.52, "0.649", fontsize=7, color=INK,
            ha="center", va="top")
    ax.text(0.657, ROW_S, "(best ckpt per arm,\nall arms)",
            fontsize=7, color=GRAY, ha="left", va="center", linespacing=1.2)

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(-0.9, 1.75)
    ax.set_xticks([0.58, 0.62, 0.66, 0.70, 0.74])
    ax.set_yticks([ROW_T, ROW_S])
    ax.set_yticklabels(["teacher", "all students"], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Cross-seed diversity (pairwise LPIPS, 8 seeds)")
    ax.set_title("Survives: cross-seed diversity collapse", loc="left",
                 fontsize=8, pad=8, fontweight="bold")
    nonzero_note(ax, axis="x", loc=(1.0, -0.42))


def main():
    apply_style()
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(FULL_W, 2.5),
        gridspec_kw={"width_ratios": [0.86, 1.0], "wspace": 0.55})
    left_panel(ax_l)
    right_panel(ax_r)
    fig.subplots_adjust(left=0.01, right=0.985, top=0.87, bottom=0.24)
    save(fig, "fig5_attribution", OUT)


if __name__ == "__main__":
    main()
