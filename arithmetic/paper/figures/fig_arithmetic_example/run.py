"""
fig_arithmetic_example: concrete carry-cascade example with SoRL token annotations.

Shows `959271 + 040756 = 1000027` — a 4-deep carry cascade — with:
  - the two operands and answer digits
  - subtask label at each answer position (Quirke et al. taxonomy)
  - SoRL abstraction token assigned at each position
  - carry chain highlighted with a clean bracket above the answer row

Token assignments are from model `add_sub_sorl_v1_abs30_K1_100K`
(2L/3H/510d, K=1, abs30) as recorded in the dashboard vignette.

Output: fig_arithmetic_example.pdf  (also .png at 300 dpi)

Usage:
  /opt/pytorch/bin/python3 paper/figures/fig_arithmetic_example/run.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Example data ──────────────────────────────────────────────────────────────
# 959271 + 040756 = 1000027
# Positions d0..d6 left-to-right (d0 = overflow/MSB, d6 = LSB)
EXAMPLE = {
    "a":        ["-", "9", "5", "9", "2", "7", "1"],   # d0 has no operand digit
    "b":        ["+", "0", "4", "0", "7", "5", "6"],
    "answer":   ["1", "0", "0", "0", "0", "2", "7"],
    "subtasks": ["UC", "US", "US", "US", "US", "SC", "SA"],
    "tokens":   ["t2", "t2", "t6", "t2", "t1", "t16", "t3"],
    "carry_in": [True, True, True, True, True, True, False],
}

# Subtask colours
SUBTASK_COLORS = {
    "SA": "#d4e8d4",   # light green  — trivial
    "SC": "#fde8a0",   # amber        — generates carry
    "SS": "#fdd0a0",   # light orange — sum-9
    "UC": "#c6d9f5",   # light blue   — uses carry
    "US": "#9bbfe0",   # deeper blue  — cascade
    "MD": "#d4e8d4",
    "MB": "#fde8a0",
    "ME": "#fdd0a0",
    "UB": "#c6d9f5",
    "UD": "#9bbfe0",
}
SUBTASK_FULL = {
    "SA": "SA — simple add",
    "SC": "SC — generates carry",
    "UC": "UC — uses carry (overflow)",
    "US": "US — cascade (sum-9 chain)",
}
TOKEN_COLOR  = "#e8d5f5"   # soft purple for token boxes
TOKEN_BORDER = "#9966cc"
TOKEN_TEXT   = "#5500aa"


def draw_digit_box(ax, x, y, text, color, fontsize=13, bold=False,
                   box_w=0.84, box_h=0.72):
    """Draw a rounded rectangle with centred text."""
    patch = FancyBboxPatch(
        (x - box_w / 2, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="#888", linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color="#222", zorder=3)


def draw_token_box(ax, x, y, text):
    """Draw a dashed-border rounded rectangle for a SoRL token."""
    patch = FancyBboxPatch(
        (x - 0.40, y - 0.30), 0.80, 0.60,
        boxstyle="round,pad=0.05",
        facecolor=TOKEN_COLOR, edgecolor=TOKEN_BORDER,
        linewidth=0.9, linestyle="--", zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=10, color=TOKEN_TEXT, fontweight="bold", zorder=3)


def draw_carry_bracket(ax, x_left, x_right, y_top, label="carry cascade"):
    """
    Draw a horizontal bracket above the answer row spanning x_left..x_right.
    The bracket consists of: left tick | flat bar | right tick, then a label above.
    """
    tick_h = 0.13   # height of vertical ticks
    lw     = 1.4
    color  = "#cc4400"

    # left tick
    ax.plot([x_left,  x_left],  [y_top - tick_h, y_top], color=color, lw=lw, zorder=4)
    # right tick
    ax.plot([x_right, x_right], [y_top - tick_h, y_top], color=color, lw=lw, zorder=4)
    # horizontal bar
    ax.plot([x_left,  x_right], [y_top, y_top],           color=color, lw=lw, zorder=4)
    # label centred above bar
    ax.text((x_left + x_right) / 2, y_top + 0.08, label,
            ha="center", va="bottom", fontsize=8.5,
            color=color, style="italic", zorder=4)


def draw_carry_arrows(ax, xs, carry_positions, y_mid):
    """
    Draw straight horizontal arrows between adjacent answer boxes
    at the mid-height of the answer row, pointing right-to-left (carry direction).
    carry_positions: sorted list of column indices that have carry_in=True.
    """
    for idx in range(len(carry_positions) - 1):
        i_right = carry_positions[idx + 1]   # source column (rightward)
        i_left  = carry_positions[idx]        # target column (leftward)
        x_start = xs[i_right] + 0.44         # right edge of source box
        x_end   = xs[i_left]  - 0.44         # left  edge of target box
        ax.annotate(
            "",
            xy=(x_end,   y_mid),
            xytext=(x_start, y_mid),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#cc4400",
                lw=1.4,
                mutation_scale=10,
                connectionstyle="arc3,rad=0.0",   # perfectly straight
            ),
            zorder=5,
        )


def main():
    n_cols = len(EXAMPLE["answer"])   # 7  (d0..d6)

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # coordinate system: x = column index (0..6), y = row height
    # rows (bottom to top): token=0.7, subtask=1.7, answer=2.8,
    #                        divider=3.3, addend_b=3.8, addend_a=4.6
    Y_TOKEN   = 0.70
    Y_SUBTASK = 1.70
    Y_ANSWER  = 2.80
    Y_DIV     = 3.28
    Y_B       = 3.78
    Y_A       = 4.58
    Y_HEADER  = 5.05
    Y_BRACKET = 3.22   # bracket sits just above the answer boxes (below the divider)

    ax.set_xlim(-1.35, n_cols - 0.4)
    ax.set_ylim(0.05, 5.55)
    ax.axis("off")

    xs = list(range(n_cols))   # x-coords for columns d0..d6

    # ── Column headers d0..d6 ────────────────────────────────────────────────
    for i, x in enumerate(xs):
        ax.text(x, Y_HEADER, f"d{i}", ha="center", va="center",
                fontsize=9, color="#999")

    # ── Row labels (left-aligned, italic) ────────────────────────────────────
    ROW_LABEL_X = -1.28
    for y, label in [
        (Y_A,       "Addend A"),
        (Y_B,       "Addend B"),
        (Y_ANSWER,  "Answer"),
        (Y_SUBTASK, "Subtask"),
        (Y_TOKEN,   "DLR token"),
    ]:
        ax.text(ROW_LABEL_X, y, label,
                ha="left", va="center",
                fontsize=10, color="#444", style="italic")

    # ── Addend A digits (columns d1..d6) ─────────────────────────────────────
    for i, digit in enumerate(EXAMPLE["a"]):
        if digit == "-":
            continue   # d0 has no operand digit for A
        ax.text(xs[i], Y_A, digit,
                ha="center", va="center", fontsize=13, color="#333")

    # ── Addend B digits + leading "+" ────────────────────────────────────────
    ax.text(xs[1] - 0.82, Y_B, "+",
            ha="center", va="center", fontsize=14, color="#333", fontweight="bold")
    for i, digit in enumerate(EXAMPLE["b"]):
        if digit == "+":
            continue
        ax.text(xs[i], Y_B, digit,
                ha="center", va="center", fontsize=13, color="#333")

    # ── Horizontal dividing line ──────────────────────────────────────────────
    line_x0 = xs[0]  - 0.48
    line_x1 = xs[-1] + 0.48
    total_w  = ax.get_xlim()[1] - ax.get_xlim()[0]
    xmin_frac = (line_x0 - ax.get_xlim()[0]) / total_w
    xmax_frac = (line_x1 - ax.get_xlim()[0]) / total_w
    ax.axhline(y=Y_DIV, xmin=xmin_frac, xmax=xmax_frac,
               color="#aaa", linewidth=1.0)

    # ── Answer digits with subtask colouring ─────────────────────────────────
    for i, (digit, sub) in enumerate(zip(EXAMPLE["answer"], EXAMPLE["subtasks"])):
        color = SUBTASK_COLORS.get(sub, "#f0f0f0")
        draw_digit_box(ax, xs[i], Y_ANSWER, digit, color,
                       fontsize=13, bold=True)

    # ── Subtask labels ────────────────────────────────────────────────────────
    for i, sub in enumerate(EXAMPLE["subtasks"]):
        color = SUBTASK_COLORS.get(sub, "#f0f0f0")
        draw_digit_box(ax, xs[i], Y_SUBTASK, sub, color, fontsize=10,
                       box_w=0.84, box_h=0.68)

    # ── SoRL token boxes ─────────────────────────────────────────────────────
    for i, tok in enumerate(EXAMPLE["tokens"]):
        draw_token_box(ax, xs[i], Y_TOKEN, tok)

    # ── Carry bracket above answer row ───────────────────────────────────────
    carry_cols = [i for i, c in enumerate(EXAMPLE["carry_in"]) if c]
    # bracket spans leftmost (d0) to rightmost carry column (d5)
    bracket_left  = xs[carry_cols[0]]  - 0.44
    bracket_right = xs[carry_cols[-1]] + 0.44
    draw_carry_bracket(ax, bracket_left, bracket_right,
                       y_top=Y_BRACKET, label="carry cascade")

    # ── Carry arrows between answer boxes ────────────────────────────────────
    draw_carry_arrows(ax, xs, carry_cols, y_mid=Y_ANSWER)

    # ── Compact text legend at bottom ────────────────────────────────────────
    legend_text = (
        "SA — simple add   ·   SC — generates carry   ·   "
        "UC — uses carry   ·   US — cascade     "
        "  [ dashed border = DLR token ]"
    )
    ax.text(0.5, -0.04, legend_text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8.5, color="#555")

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        r"$959{,}271 + 040{,}756 = 1{,}000{,}027$"
        "  —  4-deep carry cascade",
        fontsize=13, pad=10, color="#222",
    )

    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, "fig_arithmetic_example.pdf")
    png_path = os.path.join(OUT_DIR, "fig_arithmetic_example.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
