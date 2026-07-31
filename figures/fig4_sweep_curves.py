"""F4: full-sweep aesthetic trajectories — relay (5 ckpts) vs direct (10 ckpts).

Every plotted number is transcribed verbatim from frozen local sources:
  relay student W7 (500-2500):
      experiments/results/2026-07-14-e0-full-table-g1.md, E0 q150 table lines 11-15 (aes col);
      cross-checked against 2026-07-25 note line 19 (@500 0.577 / @1000 0.559)
  direct student E1a (500-5000, single-seed sweep, pattern reference only):
      experiments/results/acceptance-log.md, row #12 (line 25), parenthesized
      E1a aes full trajectory "@500 0.5243/.../@5000 0.5196"
  quantitative-optimum band 500-1000 and subjective pick @2500:
      2026-07-14-e0-full-table-g1.md lines 28 ("质量最优在 @500-@1000") and
      15 ("@2500(肉眼最佳)")
"""
import matplotlib.pyplot as plt
from figstyle import (NAVY, NAVY_SOFT, ORANGE, GRAY, INK,
                      FULL_W, apply_style, save, ygrid, nonzero_note)

OUT = "/Users/x3y/Desktop/ChenQingzhan-DMD-distillation/figures"

RELAY_X = [500, 1000, 1500, 2000, 2500]
# 2026-07-14-e0-full-table-g1.md L11-L15
RELAY_Y = [0.5768, 0.5592, 0.5433, 0.5477, 0.5379]

DIRECT_X = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# acceptance-log.md L25 (row #12)
DIRECT_Y = [0.5243, 0.5665, 0.5523, 0.5609, 0.5385,
            0.5219, 0.5327, 0.5281, 0.5226, 0.5196]


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(FULL_W, 2.8))

    # quantitative-optimum region (light orange background band, 500-1000)
    ax.axvspan(500, 1000, color=ORANGE, alpha=0.10, lw=0, zorder=0)
    ax.text(750, 0.5875, "quantitative optimum", fontsize=7, color=ORANGE,
            ha="center", va="bottom", clip_on=False)

    ax.plot(DIRECT_X, DIRECT_Y, color=NAVY, lw=1.5, marker="o", ms=3.3,
            zorder=5, clip_on=False)
    ax.plot(RELAY_X, RELAY_Y, color=NAVY_SOFT, lw=1.5, ls=(0, (5, 2.2)),
            marker="^", ms=3.8, zorder=4, clip_on=False)

    # relay endpoint: orange open ring = subjective pick (warning highlight)
    ax.plot([2500], [RELAY_Y[-1]], marker="o", ms=7.5, mfc="none",
            mec=ORANGE, mew=1.2, zorder=6, clip_on=False)

    # ---- key-point labels (first/last values and peaks only) ----
    ax.text(500, 0.5768 + 0.0035, "0.577", fontsize=7, color=INK,
            ha="center", va="bottom")
    ax.annotate("peak 0.567", xy=(1000, 0.5665), xytext=(1210, 0.5723),
                fontsize=7, color=INK, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6,
                                shrinkA=1, shrinkB=3))
    ax.text(500, 0.5243 - 0.0038, "0.524", fontsize=7, color=INK,
            ha="center", va="top")

    # relay line-end block
    ax.text(2580, 0.5445, "relay student · end 0.538", fontsize=7.5,
            color=INK, ha="left", va="bottom", fontweight="bold")
    ax.text(2580, 0.5425, "subjective pick (end of decline)", fontsize=7,
            color=ORANGE, ha="left", va="top")

    # direct line-end block
    ax.text(5090, 0.5196 + 0.0028, "direct student (primary)", fontsize=7.5,
            color=INK, ha="left", va="bottom", fontweight="bold")
    ax.text(5090, 0.5178, "end 0.520", fontsize=7, color=GRAY,
            ha="left", va="top")

    ax.set_xlim(300, 6650)
    ax.set_ylim(0.510, 0.586)
    ax.set_xticks([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
    ax.set_yticks([0.51, 0.53, 0.55, 0.57])
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Aesthetic quality\n(150 standard prompts)")
    ax.spines["left"].set_bounds(0.51, 0.58)
    ax.spines["bottom"].set_bounds(300, 5000)
    ygrid(ax)
    nonzero_note(ax)

    save(fig, "fig4_sweep_curves", OUT)


if __name__ == "__main__":
    main()
