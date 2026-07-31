"""F3: degradation audit panels — dynamic degree not degraded (a),
cross-seed diversity consistently lower (b).

Every plotted number is transcribed verbatim from frozen local sources
(final arbiter = 2026-07-20 G2 table; cross-checked against 2026-07-14 E0 note;
band matches report §6.1 text: every 4-step student, the ablation arms, and
the 8-step intermediate):
  teacher DD_clean 0.625, diversity 0.732:
      experiments/results/2026-07-20-g2-relay-vs-direct-final.md, table line 12
  student champion ckpts (best-of-sweep per arm), same table lines 13-17:
      E1a@1000  DD 0.750  div 0.635
      W7@500    DD 0.825  div 0.598
      W7@1000   DD 1.000  div 0.613
      E1b@500   DD 0.975  div 0.628
      W1@1000   DD 0.950  div 0.649
  ablation arms + 8-step intermediate (revision 2026-07-28):
      E2a@2000 (GAN-off champion)      div 0.5860  — 2026-07-24 note L12
      E2b@500  (indep-pairing champion) div 0.590   — 2026-07-25 note L9
      W5@2500  (8-step intermediate)   DD 0.950  div 0.5949 — 2026-07-14 L19/L75
  motion-smoothness footnote (0.97+): 2026-07-14-e0-full-table-g1.md line 79
      ("motion_smoothness 全员 0.970-0.987")
"""
import matplotlib.pyplot as plt
from figstyle import (NAVY, NAVY_SOFT, ORANGE, GRAY, INK, ORANGE_TINT,
                      FULL_W, apply_style, save, nonzero_note, axis_break)

OUT = "/Users/x3y/Desktop/ChenQingzhan-DMD-distillation/figures"

TEACHER_DD = 0.625          # 07-20 L12
STUDENT_DD = [0.750, 0.825, 1.000, 0.975, 0.950,   # 07-20 L13-17
              0.950]                               # W5@2500, 07-14 L75
TEACHER_DIV = 0.732         # 07-20 L12
STUDENT_DIV = [0.635, 0.598, 0.613, 0.628, 0.649,  # 07-20 L13-17
               0.5860, 0.590, 0.5949]              # E2a 07-24 L12 / E2b 07-25 L9 / W5 07-14 L19

ROW_T, ROW_S = 1.0, 0.0
BAR_H = 0.42


def band(ax, lo, hi, members, face, edge, tick_color):
    ax.barh(ROW_S, hi - lo, left=lo, height=BAR_H, color=face,
            edgecolor=edge, linewidth=0.7, zorder=3)
    for m in members:
        ax.plot([m, m], [ROW_S - 0.30 * BAR_H, ROW_S + 0.30 * BAR_H],
                color=tick_color, lw=0.9, zorder=4)


def rowsetup(ax):
    ax.set_yticks([ROW_T, ROW_S])
    ax.set_yticklabels(["teacher", "all students"], fontsize=8)
    ax.set_ylim(-0.9, 1.75)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def main():
    apply_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FULL_W, 2.35), gridspec_kw={"wspace": 0.42})

    # ---------------- (a) dynamic degree: zero-origin axis ----------------
    ax_a.barh(ROW_T, TEACHER_DD, height=BAR_H, color=NAVY, zorder=3)
    ax_a.text(TEACHER_DD + 0.02, ROW_T, "0.625", fontsize=7, color=INK,
              ha="left", va="center")
    band(ax_a, min(STUDENT_DD), max(STUDENT_DD), STUDENT_DD,
         face=NAVY_SOFT, edge=NAVY, tick_color=NAVY)
    ax_a.text(min(STUDENT_DD), ROW_S - 0.52, "0.75", fontsize=7, color=INK,
              ha="center", va="top")
    ax_a.text(max(STUDENT_DD), ROW_S - 0.52, "1.00", fontsize=7, color=INK,
              ha="center", va="top")
    ax_a.text(0.715, ROW_S, "(best ckpt per arm)", fontsize=7, color=GRAY,
              ha="right", va="center")
    ax_a.annotate("higher, not lower", xy=(0.685, 0.55), fontsize=7,
                  color=GRAY, ha="center", va="center", style="italic")
    ax_a.set_xlim(0, 1.06)
    ax_a.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ax_a.set_xlabel("Dynamic degree")
    ax_a.set_title("(a) Dynamic degree (40 motion prompts): not degraded",
                   loc="left", fontsize=8, pad=8)
    rowsetup(ax_a)
    fig.text(0.065, 0.015, "motion smoothness 0.97+: high dynamics reflect "
             "real motion, not flicker", fontsize=7, color=GRAY)

    # ------------- (b) diversity: magnified non-zero axis -----------------
    XLO, XHI = 0.57, 0.755
    ax_b.barh(ROW_T, TEACHER_DIV - XLO, left=XLO, height=BAR_H, color=NAVY,
              zorder=3)
    ax_b.plot([TEACHER_DIV], [ROW_T + 0.42], marker="v", ms=5, color=NAVY,
              zorder=5, clip_on=False)
    ax_b.text(TEACHER_DIV - 0.008, ROW_T + 0.45, "teacher 0.732", fontsize=7,
              color=INK, ha="right", va="center")
    axis_break(ax_b, XLO + 0.006, ROW_T, BAR_H)
    # ORANGE band: the one orange data highlight = main degradation axis
    band(ax_b, min(STUDENT_DIV), max(STUDENT_DIV), STUDENT_DIV,
         face=ORANGE_TINT, edge=ORANGE, tick_color=ORANGE)
    ax_b.text(min(STUDENT_DIV), ROW_S - 0.52, "0.586", fontsize=7, color=INK,
              ha="center", va="top")
    ax_b.text(max(STUDENT_DIV), ROW_S - 0.52, "0.649", fontsize=7, color=INK,
              ha="center", va="top")
    ax_b.text(0.657, ROW_S, "(best ckpt per arm, all arms)", fontsize=7,
              color=GRAY, ha="left", va="center")
    ax_b.set_xlim(XLO, XHI)
    ax_b.set_xticks([0.58, 0.62, 0.66, 0.70, 0.74])
    ax_b.set_xlabel("Cross-seed diversity (pairwise LPIPS, 8 seeds)")
    ax_b.set_title("(b) Cross-seed diversity (LPIPS): consistently lower",
                   loc="left", fontsize=8, pad=8)
    rowsetup(ax_b)
    ax_b.annotate("higher = more diverse →", xy=(1.0, -0.315),
                  xycoords="axes fraction", fontsize=7, color=GRAY,
                  ha="right", va="top", annotation_clip=False)
    nonzero_note(ax_b, axis="x", loc=(1.0, -0.44))

    fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.31)
    save(fig, "fig3_dd_diversity_panels", OUT)


if __name__ == "__main__":
    main()
