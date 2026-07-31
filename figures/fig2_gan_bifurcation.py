"""F2: GAN bifurcation — aesthetic quality vs training iteration, three arms.

Every plotted number is transcribed verbatim from frozen local result notes:
  E2a (GAN off)        experiments/results/2026-07-24-e2a-fulltable-ch3.md, table lines 9-13 (aes col)
  E2b (GAN on, indep)  experiments/results/2026-07-25-e2b-fulltable-ch3-threearm.md, table lines 9-13 (aes col)
  W7  (GAN on, shared) experiments/results/2026-07-14-e0-full-table-g1.md, E0 q150 table lines 11-15 (aes col);
                       cross-checked against 2026-07-25 note line 19 ("@500 aes 0.577 ... @1000 0.559 ...")
All three arms start from the same 8-step intermediate (W5) initialization
(single-variable discipline verified in acceptance-log.md row #11).
"""
import matplotlib.pyplot as plt
from figstyle import (NAVY, NAVY_SOFT, ORANGE, TEAL, GRAY, INK,
                      FULL_W, apply_style, save, ygrid, nonzero_note)

OUT = "/Users/x3y/Desktop/ChenQingzhan-DMD-distillation/figures"

ITERS = [500, 1000, 1500, 2000, 2500]
# 2026-07-24-e2a-fulltable-ch3.md L9-L13
E2A = [0.5908, 0.5921, 0.5984, 0.6109, 0.6074]
# 2026-07-25-e2b-fulltable-ch3-threearm.md L9-L13
E2B = [0.5774, 0.5670, 0.5477, 0.5471, 0.5487]
# 2026-07-14-e0-full-table-g1.md L11-L15 (W7 @500..@2500)
W7 = [0.5768, 0.5592, 0.5433, 0.5477, 0.5379]


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W, 3.0))

    ax.plot(ITERS, E2A, color=TEAL, lw=1.5, marker="o", ms=3.6,
            zorder=5, clip_on=False)
    ax.plot(ITERS, E2B, color=NAVY, lw=1.5, marker="s", ms=3.3,
            zorder=4, clip_on=False)
    ax.plot(ITERS, W7, color=NAVY_SOFT, lw=1.5, ls=(0, (5, 2.2)), marker="^",
            ms=3.8, zorder=4, clip_on=False)

    # ---- line-end labels (identity not by color alone) ----
    def endlabel(y, name, val, dy_name=0.0, dy_val=0.0):
        ax.text(2585, y + 0.0042 + dy_name, name, fontsize=7.5, color=INK,
                ha="left", va="bottom", fontweight="bold")
        ax.text(2585, y - 0.0012 + dy_val, f"end {val}", fontsize=7,
                color=GRAY, ha="left", va="top")

    endlabel(E2A[-1], "GAN off", "0.607")
    endlabel(E2B[-1], "GAN on · independent (t,ε)", "0.549",
             dy_name=0.0022, dy_val=0.0022)
    endlabel(W7[-1], "GAN on · shared (t,ε)", "0.538",
             dy_name=-0.0035, dy_val=-0.0035)

    # peak of the GAN-off arm (0.6109 @2000, source line 12)
    ax.annotate("peak 0.611", xy=(2000, 0.6109), xytext=(1730, 0.6165),
                fontsize=7, color=TEAL, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.6,
                                shrinkA=1, shrinkB=3))

    # shared starting point: all arms initialized from the 8-step intermediate
    ax.plot([440, 440], [W7[0] - 0.0015, E2A[0] + 0.0015], color=GRAY,
            lw=0.8, clip_on=False)
    ax.text(455, 0.5985, "same init\n(8-step intermediate)", fontsize=7,
            color=GRAY, ha="left", va="bottom", linespacing=1.25)

    # single orange accent: warning about the GAN-on trend (not a data series)
    ax.text(1450, 0.5655, "quality declines\nwith training", fontsize=7,
            color=ORANGE, ha="left", va="center", linespacing=1.25)
    ax.annotate("", xy=(1720, 0.5515), xytext=(1465, 0.5595),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8))

    ax.set_xlim(370, 3520)
    ax.set_ylim(0.530, 0.622)
    ax.set_xticks(ITERS)
    ax.set_yticks([0.54, 0.56, 0.58, 0.60, 0.62])
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Aesthetic quality (150 standard prompts)")
    ax.spines["left"].set_bounds(0.53, 0.62)
    ax.spines["bottom"].set_bounds(370, 2500)
    ygrid(ax)
    nonzero_note(ax)

    save(fig, "fig2_gan_bifurcation", OUT)


if __name__ == "__main__":
    main()
