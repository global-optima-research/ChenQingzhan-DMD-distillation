"""F1: step-count relay vs matched-budget direct distillation (design schematic).

No data axes. Structural facts transcribed from frozen local sources:
  research/thesis_ch2_draft.md §2.1
    L12: relay arm  W5 8-step 2500 iter -> W7 4-step 2500 iter,
         "仅继承 W5@2500 生成器权重;优化器/fake score/判别器重置" (generator
         weights only; optimizer / fake score / discriminator reset)
    L13: direct arms x2, each 5000 iter; E1a LR 5e-6 (low), E1b LR 1e-5 (high)
    L14: invariants = data, 4-step t_list, discriminator structure +
         hyperparameters, evaluation protocol (matched budget per L12/L13)
  experiments/results/2026-07-20-g2-relay-vs-direct-final.md L25:
    pre-registered parity expectation; observed: direct slightly ahead
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import (NAVY, NAVY_SOFT, ORANGE, GRAY, INK, NAVY_TINT,
                      FULL_W, apply_style, save)

OUT = "/Users/x3y/Desktop/ChenQingzhan-DMD-distillation/figures"


def box(ax, x0, y0, w, h, fc, ec, lw=0.9, r=2.5, z=3):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, clip_on=False))


def arrow(ax, p0, p1, rad=0.0, color=NAVY, lw=1.1, z=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad:+.2f}", arrowstyle="-|>",
        mutation_scale=9, color=color, lw=lw, shrinkA=1, shrinkB=1,
        zorder=z, clip_on=False))


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W, 2.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- teacher node ----------------------------------------------------
    box(ax, 1, 54, 15.5, 24, fc=NAVY, ec=NAVY)
    ax.text(8.75, 68.5, "Teacher", fontsize=8, color="white", ha="center",
            va="center", fontweight="bold")
    ax.text(8.75, 60.5, "50-step, CFG", fontsize=7, color="#D8E0EE",
            ha="center", va="center")

    # ---- relay arm (upper branch) ---------------------------------------
    RY, RH = 76, 20                       # box bottom y, height
    box(ax, 25, RY, 26, RH, fc="white", ec=NAVY)
    ax.text(38, RY + 13.5, "8-step intermediate\nstudent", fontsize=7.5,
            color=INK, ha="center", va="center", fontweight="bold",
            linespacing=1.15)
    ax.text(38, RY + 4, "2500 iters", fontsize=7, color=GRAY,
            ha="center", va="center")

    box(ax, 72, RY, 26, RH, fc="white", ec=NAVY)
    ax.text(85, RY + 13.5, "4-step relay\nstudent", fontsize=7.5, color=INK,
            ha="center", va="center", fontweight="bold", linespacing=1.15)
    ax.text(85, RY + 4, "2500 iters", fontsize=7, color=GRAY,
            ha="center", va="center")

    # re-init chip on the connecting arrow (single orange highlight)
    arrow(ax, (51, RY + RH / 2), (72, RY + RH / 2))
    box(ax, 53.5, RY + 3.5, 16, 13, fc="white", ec=ORANGE, lw=1.0, r=2, z=4)
    ax.text(61.5, RY + 10, "↺ reset all\nbut generator", fontsize=6.4,
            color=ORANGE, ha="center", va="center", fontweight="bold",
            linespacing=1.2, zorder=5)
    ax.text(61.5, RY - 1.5, "generator weights only;\noptimizer / fake score / "
            "discriminator reset", fontsize=7, color=GRAY, ha="center",
            va="top", linespacing=1.25, zorder=5)

    # ---- direct arm (lower branch) --------------------------------------
    DY, DH = 30, 20
    box(ax, 24, DY, 37, DH, fc="white", ec=NAVY)
    ax.text(42.5, DY + 13.5, "4-step direct student ×2 arms", fontsize=7.2,
            color=INK, ha="center", va="center", fontweight="bold")
    ax.text(42.5, DY + 5, "5000 iters each · low / high LR", fontsize=6.9,
            color=GRAY, ha="center", va="center")

    # branch arrows from teacher
    arrow(ax, (16.5, 72), (25, RY + RH / 2), rad=-0.18)
    arrow(ax, (16.5, 60), (25, DY + DH / 2), rad=0.18)

    # ---- pre-registered / observed double badge -------------------------
    bx, bw = 62.5, 35.5
    box(ax, bx, DY + 11, bw, 10, fc="white", ec=NAVY_SOFT, lw=0.9, r=2)
    ax.text(bx + bw / 2, DY + 16, "pre-registered: parity expected",
            fontsize=6.8, color=INK, ha="center", va="center")
    box(ax, bx, DY - 1, bw, 10, fc=NAVY_TINT, ec=NAVY, lw=0.9, r=2)
    ax.text(bx + bw / 2, DY + 4, "observed: direct slightly ahead",
            fontsize=6.8, color=NAVY, ha="center", va="center",
            fontweight="bold")
    arrow(ax, (bx + bw / 2, DY + 11), (bx + bw / 2, DY + 9), color=GRAY,
          lw=0.9)

    # ---- invariants bar --------------------------------------------------
    box(ax, 1, 2, 97, 13, fc=NAVY_TINT, ec=NAVY, lw=0.8, r=2)
    ax.text(49.5, 8.5, "Matched:  data · budget · 4-step timestep list · "
            "discriminator · evaluation protocol", fontsize=7.3, color=NAVY,
            ha="center", va="center")

    save(fig, "fig1_relay_design", OUT)


if __name__ == "__main__":
    main()
