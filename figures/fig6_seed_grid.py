"""F6: teacher vs 4-step student, same prompt, 8 seeds each (frame grid).

Assets (16 PNG frames, 832x480 each) from slides/covers/:
  top row:    p3v2_teacher_s0.png ... p3v2_teacher_s7.png   (seeds 0-7)
  bottom row: p3v2_e1a_s{0,2,3,4,6,8,9,11}.png              (in that order)
Seed tag in each cell = the real seed number from the filename.
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from PIL import Image
from figstyle import INK, FULL_W, apply_style, save

OUT = "/Users/x3y/Desktop/ChenQingzhan-DMD-distillation/figures"
SRC = "/Users/x3y/Desktop/ChenQingzhan-DMD-distillation/slides/covers"

TOP = [f"p3v2_teacher_s{i}.png" for i in range(8)]          # s0..s7
BOT_SEEDS = [0, 2, 3, 4, 6, 8, 9, 11]
BOT = [f"p3v2_e1a_s{i}.png" for i in BOT_SEEDS]

LABEL_TOP = "Teacher (50-step, CFG) — 8 seeds, 8 compositions"
LABEL_BOT = "4-step student — 8 seeds, near-identical composition"

N = 8
GAP = 0.009          # inches (~2.7 px at 300 dpi)
LABEL_H = 0.13       # inches per row label line
AR = 480 / 832       # frame aspect ratio


def main():
    apply_style()
    cell_w = (FULL_W - (N - 1) * GAP) / N
    cell_h = cell_w * AR
    fig_h = 2 * (LABEL_H + cell_h) + GAP
    fig = plt.figure(figsize=(FULL_W, fig_h))

    def add_row(files, seeds, label, y_img):
        fig.text(0.002, (y_img + cell_h + 0.025) / fig_h, label,
                 fontsize=7, color=INK, ha="left", va="bottom",
                 fontweight="bold")
        for k, (f, s) in enumerate(zip(files, seeds)):
            x = k * (cell_w + GAP)
            ax = fig.add_axes([x / FULL_W, y_img / fig_h,
                               cell_w / FULL_W, cell_h / fig_h])
            im = Image.open(f"{SRC}/{f}")
            im.thumbnail((484, 280), Image.LANCZOS)  # ~2x print resolution
            ax.imshow(im)
            ax.set_axis_off()
            ax.text(0.965, 0.06, f"s{s}", transform=ax.transAxes, fontsize=7,
                    color="white", ha="right", va="bottom",
                    path_effects=[pe.withStroke(linewidth=1.4,
                                                foreground="#000000")])

    # stack, top to bottom: top label / top row / gap / bottom label / bottom row
    add_row(TOP, list(range(8)), LABEL_TOP, y_img=cell_h + LABEL_H + GAP)
    add_row(BOT, BOT_SEEDS, LABEL_BOT, y_img=0.0)
    save(fig, "fig6_seed_grid", OUT)


if __name__ == "__main__":
    main()
