#!/usr/bin/env python3
"""Generate the recursive KNN exploration diagram used as Figure 1."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


MPL_CACHE = Path(tempfile.gettempdir()) / "rtknnc-matplotlib-cache"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]

NAVY = "#1f4e79"
ORANGE = "#d97904"
TEAL = "#238b7e"
LIGHT_BLUE = "#8ecae6"
GREY = "#9aa0a6"
LIGHT_GREY = "#d8dde3"
PURPLE = "#7a5195"


def node(
    axis,
    xy: tuple[float, float],
    label: str,
    *,
    face: str,
    edge: str,
    radius: float = 0.28,
    linewidth: float = 2.0,
    text_colour: str = "black",
    zorder: int = 4,
) -> None:
    axis.add_patch(
        Circle(
            xy,
            radius,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            zorder=zorder,
        )
    )
    axis.text(
        *xy,
        label,
        ha="center",
        va="center",
        fontsize=12,
        color=text_colour,
        zorder=zorder + 1,
    )


def arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str,
    linewidth: float = 2.1,
    linestyle: str = "solid",
    alpha: float = 1.0,
    mutation_scale: float = 13.0,
    zorder: int = 2,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            color=colour,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            mutation_scale=mutation_scale,
            shrinkA=13,
            shrinkB=13,
            zorder=zorder,
        )
    )


def panel(axis, x0: float, title: str) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x0, 0.85),
            4.55,
            3.8,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="#fbfcfe",
            edgecolor=LIGHT_GREY,
            linewidth=1.2,
            zorder=0,
        )
    )
    axis.text(
        x0 + 2.275,
        4.42,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#25313c",
    )


def draw_figure(output_base: Path, dpi: int) -> None:
    fig, axis = plt.subplots(figsize=(15.2, 5.45))
    axis.set_xlim(0.0, 15.2)
    axis.set_ylim(0.0, 5.35)
    axis.axis("off")

    panel(axis, 0.15, r"(a) $q$ selects a visible $K$-neighbour")
    panel(axis, 5.325, r"(b) $p$ becomes the next query")
    panel(axis, 10.5, r"(c) Repeat until saturation")

    # Panel (a): q has a directed KNN list, but only the emphasized visible
    # transition is followed in this illustrative branch.
    q = (0.95, 2.65)
    p = (3.65, 2.65)
    q_other_1 = (3.25, 3.75)
    q_other_2 = (3.25, 1.55)
    axis.add_patch(
        Ellipse(
            (3.45, 2.65),
            1.75,
            3.0,
            facecolor="#f4f6f8",
            edgecolor=GREY,
            linewidth=1.1,
            linestyle=(0, (4, 4)),
            zorder=1,
        )
    )
    axis.text(3.45, 1.02, r"$\mathcal{N}_K^+(q)$", ha="center", fontsize=11)
    arrow(axis, q, q_other_1, colour=GREY, linewidth=1.2, linestyle="dashed", alpha=0.6)
    arrow(axis, q, q_other_2, colour=GREY, linewidth=1.2, linestyle="dashed", alpha=0.6)
    arrow(axis, q, p, colour=ORANGE, linewidth=2.8)
    node(axis, q, r"$q$", face=NAVY, edge=NAVY, text_colour="white")
    node(axis, p, r"$p$", face="#ffd08a", edge=ORANGE)
    node(axis, q_other_1, r"$a$", face="white", edge=GREY, linewidth=1.4)
    node(axis, q_other_2, r"$b$", face="white", edge=GREY, linewidth=1.4)
    axis.text(0.95, 1.95, "query", ha="center", color=NAVY, fontsize=10)
    axis.text(2.18, 2.84, "accepted transition", ha="center", color=ORANGE, fontsize=10)
    axis.text(3.65, 2.05, "selected neighbour", ha="center", color=ORANGE, fontsize=10)

    # A short inter-panel arrow makes the recursive call explicit.
    arrow(
        axis,
        (4.35, 2.65),
        (5.72, 2.65),
        colour=NAVY,
        linewidth=1.8,
        mutation_scale=12,
    )
    axis.text(
        5.03,
        2.93,
        r"recursive call on $p$",
        ha="center",
        fontsize=9.5,
        color=NAVY,
    )

    # Panel (b): p is no longer merely a neighbour; it is now a query whose
    # own outgoing neighbourhood is exposed.  A reverse-visible predecessor
    # is included discreetly because the implementation explores both lists.
    p2 = (6.15, 2.65)
    r = (8.65, 3.65)
    s = (9.15, 2.55)
    t = (8.55, 1.45)
    u = (6.55, 3.75)
    axis.add_patch(
        Ellipse(
            (8.78, 2.55),
            1.75,
            3.0,
            facecolor="#eef8f6",
            edgecolor=TEAL,
            linewidth=1.1,
            linestyle=(0, (4, 4)),
            zorder=1,
        )
    )
    axis.text(8.78, 1.02, r"$\mathcal{N}_K^+(p)$", ha="center", fontsize=11)
    arrow(axis, p2, r, colour=TEAL, linewidth=2.4)
    arrow(axis, p2, s, colour=TEAL, linewidth=2.4)
    arrow(axis, p2, t, colour=GREY, linewidth=1.2, linestyle="dashed", alpha=0.65)
    arrow(axis, u, p2, colour=PURPLE, linewidth=1.5, linestyle="dashed", alpha=0.8)
    node(axis, p2, r"$p$", face="#ffd08a", edge=ORANGE)
    node(axis, r, r"$r$", face="#9dd9d1", edge=TEAL)
    node(axis, s, r"$s$", face="#9dd9d1", edge=TEAL)
    node(axis, t, r"$t$", face="white", edge=GREY, linewidth=1.4)
    node(axis, u, r"$u$", face="white", edge=PURPLE, radius=0.22, linewidth=1.4)
    axis.text(6.15, 1.95, "new query", ha="center", color=ORANGE, fontsize=10)
    axis.text(7.5, 3.92, "reveals its own KNN", ha="center", color=TEAL, fontsize=10)
    axis.text(
        6.08,
        4.08,
        r"$u\in\mathcal{V}_K^-(p)$",
        ha="center",
        fontsize=8.7,
        color=PURPLE,
    )

    # Panel (c): compact recursion tree.  Colours denote discovery waves, not
    # cluster identities.
    q3 = (11.05, 2.65)
    p3 = (12.05, 2.65)
    r3 = (13.05, 3.45)
    s3 = (13.05, 1.85)
    v3 = (14.15, 3.95)
    w3 = (14.15, 3.0)
    x3 = (14.15, 2.15)
    y3 = (14.15, 1.2)
    for start, end, colour in (
        (q3, p3, ORANGE),
        (p3, r3, TEAL),
        (p3, s3, TEAL),
        (r3, v3, LIGHT_BLUE),
        (r3, w3, LIGHT_BLUE),
        (s3, x3, LIGHT_BLUE),
        (s3, y3, LIGHT_BLUE),
    ):
        arrow(axis, start, end, colour=colour, linewidth=1.8, mutation_scale=11)
    node(axis, q3, r"$q$", face=NAVY, edge=NAVY, text_colour="white", radius=0.24)
    node(axis, p3, r"$p$", face="#ffd08a", edge=ORANGE, radius=0.24)
    node(axis, r3, r"$r$", face="#9dd9d1", edge=TEAL, radius=0.24)
    node(axis, s3, r"$s$", face="#9dd9d1", edge=TEAL, radius=0.24)
    for xy, label in ((v3, r"$v$"), (w3, r"$w$"), (x3, r"$x$"), (y3, r"$y$")):
        node(axis, xy, label, face="#d9eff8", edge="#4b9fc0", radius=0.22, linewidth=1.5)
    axis.text(11.05, 1.15, r"$t=0$", ha="center", fontsize=9.5, color=NAVY)
    axis.text(12.05, 1.15, r"$t=1$", ha="center", fontsize=9.5, color=ORANGE)
    axis.text(13.05, 1.15, r"$t=2$", ha="center", fontsize=9.5, color=TEAL)
    axis.text(14.15, 0.91, r"$t=3,\ldots$", ha="center", fontsize=9.5, color="#4b9fc0")
    axis.text(
        12.78,
        4.05,
        "every newly reached point\nbecomes another query",
        ha="center",
        va="center",
        fontsize=9.7,
        color="#25313c",
    )

    fig.suptitle(
        "Recursive round-trip exploration on a directed KNN graph",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    axis.text(
        7.6,
        0.34,
        (
            r"$C^{t+1}=C^t\cup\!\!\bigcup_{z\in C^t}"
            r"\left(\mathcal{V}_K^+(z)\cup\mathcal{V}_K^-(z)\right)$"
            r"$\qquad$ stop when $C^{t+1}=C^t$"
        ),
        ha="center",
        va="center",
        fontsize=12,
        color="#25313c",
    )

    legend = [
        Line2D([0], [0], color=TEAL, linewidth=2.4, label="accepted visible KNN edge"),
        Line2D([0], [0], color=GREY, linewidth=1.3, linestyle="--", label="KNN edge not followed"),
        Line2D([0], [0], color=PURPLE, linewidth=1.5, linestyle="--", label="visible reverse-neighbour edge"),
    ]
    axis.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            output_base.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=ROOT / "paper" / "figure1_roundtrip_reverse_neighbor",
        help="output path without .pdf/.png suffix",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_base = arguments.output_base.resolve()
    draw_figure(output_base, arguments.dpi)
    print(f"Figure written to: {output_base}.pdf")
    print(f"Figure written to: {output_base}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
