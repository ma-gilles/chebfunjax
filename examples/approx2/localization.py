"""Low-rank approximation and localized singularities.

Translation of approx2/Localization.m (Trefethen, 2016):
low-rank compression is dramatic when a (near-)singularity is
localized — a sharp spike inside the domain, or a real singularity
just outside a corner — shown by rank vs. length at eps = 1e-10,
with the GE pivot-cross pictures.

Original: https://www.chebfun.org/examples/approx2/Localization.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import hashlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

PUBLISH_DPI = 5630 * 0.0254
TRUE_PIXEL_FONT = 13.3333
FONT_PT = TRUE_PIXEL_FONT * 72.0 / PUBLISH_DPI
matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.linewidth": 0.5,
    "axes.labelsize": FONT_PT,
    "axes.titlesize": FONT_PT,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
    "axes.grid": False,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.6,
    "savefig.bbox": None,
    "savefig.facecolor": "white",
    "savefig.dpi": PUBLISH_DPI,
})

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')
_CHECKPOINTS = os.path.join(_HERE, 'localization-checkpoints')

EP = 1e-10


def _report(F):
    m, n = F.length()
    print("r =")
    print(f"{int(F.rank):6d}")
    print("m =")
    print(f"{m:6d}")
    print("n =")
    print(f"{n:6d}")


def _write_checkpoint(name, payload):
    path = Path(_CHECKPOINTS) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pivot_plot(F, k, xticks, yticks):
    piv = np.array(F.pivot_locations)
    n = piv.shape[0]
    pivots = [[float(value) for value in row] for row in piv]
    pivot_payload = {
        "qualification_baseline_head": "2208e003dcd9266ba0015f851285a4aa3b95b743",
        "figure_index": int(k),
        "source_field_case": (2 if k == 1 else 4),
        "rank": int(n),
        "pivot_locations": pivots,
        "pivot_locations_hex": [[float(value).hex() for value in row] for row in pivots],
    }
    pivot_payload["pivot_locations_sha256"] = hashlib.sha256(
        json.dumps(pivots, separators=(",", ":")).encode()
    ).hexdigest()
    _write_checkpoint(f"case{k:02d}_pivots.json", pivot_payload)
    print("n =")
    print(f"{n:6d}")
    fig, ax = plt.subplots(
        figsize=(610.0 / PUBLISH_DPI, 276.0 / PUBLISH_DPI), dpi=PUBLISH_DPI
    )
    # Keep the historical MATLAB publisher's default axes position; the source
    # uses axis square and applies no pixel offset.
    axes_left_correction_px = 0.0
    axes_position = [
        0.13 + axes_left_correction_px / 610.0,
        0.11,
        0.775,
        0.815,
    ]
    ax.set_position(axes_position)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    source_tick_formatter = FuncFormatter(lambda value, _position: f"{value:g}")
    ax.xaxis.set_major_formatter(source_tick_formatter)
    ax.yaxis.set_major_formatter(source_tick_formatter)
    artist_lines = []
    for (px, py) in piv:
        horizontal = ax.plot(
            [-1, 1], [py, py], color="black", linewidth=1.6, linestyle="-", clip_on=True
        )[0]
        vertical = ax.plot(
            [px, px], [-1, 1], color="black", linewidth=1.6, linestyle="-", clip_on=True
        )[0]
        artist_lines.extend([horizontal, vertical])
    points = ax.plot(
        piv[:, 0], piv[:, 1], linestyle="None", marker="o", color="red",
        markerfacecolor="none", markersize=8, markeredgewidth=1.6, clip_on=False,
    )[0]
    fig.set_facecolor("white")
    artist_payload = {
        "qualification_baseline_head": "2208e003dcd9266ba0015f851285a4aa3b95b743",
        "figure_index": int(k),
        "source_field_case": (2 if k == 1 else 4),
        "canvas_pixels": [610, 276],
        "dpi": PUBLISH_DPI,
        "font_true_pixels": TRUE_PIXEL_FONT,
        "font_points": FONT_PT,
        "axes_position_request": axes_position,
        "axes_left_correction_pixels": axes_left_correction_px,
        "xlim": [-1.0, 1.0],
        "ylim": [-1.0, 1.0],
        "xticks": [float(value) for value in xticks],
        "yticks": [float(value) for value in yticks],
        "formatted_xticklabels": [source_tick_formatter(float(value), i) for i, value in enumerate(xticks)],
        "formatted_yticklabels": [source_tick_formatter(float(value), i) for i, value in enumerate(yticks)],
        "aspect": "equal",
        "crosses": [
            {"x": line.get_xdata().tolist(), "y": line.get_ydata().tolist(),
             "color": "black", "linewidth_pt": 1.6, "clip_on": True}
            for line in artist_lines
        ],
        "points": {
            "x": points.get_xdata().tolist(),
            "y": points.get_ydata().tolist(),
            "color": "red",
            "marker": "o",
            "markersize_pt": 8,
            "markeredgewidth_pt": 1.6,
            "markerfacecolor": "none",
            "clip_on": False,
        },
    }
    _write_checkpoint(f"case{k:02d}_artists.json", artist_payload)
    output_path = os.path.join(_IMG, f"Localization_{k:02d}.png")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"savefig.bbox": None}):
        fig.savefig(
            output_path,
            dpi=PUBLISH_DPI,
            facecolor="white",
            bbox_inches=None,
        )
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Broad spike inside the domain: modest compression.
    f = Chebfun2.from_function(
        lambda x, y: 1 / (1 + (x - .2)**2 + (y - .5)**2), tol=EP)
    _report(f)

    # Sharp spike: dramatic difference between rank and length.
    f = Chebfun2.from_function(
        lambda x, y: 1 / (0.001 + (x - .2)**2 + (y - .5)**2), tol=EP)
    _report(f)
    _pivot_plot(f, 1, [-1, 0.2, 1], [-1, 0.5, 1])

    # Real singularity outside a corner, not very close: little
    # compression.
    g = Chebfun2.from_function(
        lambda x, y: 1 / ((x + 1.2)**2 + (y + 1.2)**2), tol=EP)
    _report(g)

    # Singularity very close to the corner: striking compression.
    g = Chebfun2.from_function(
        lambda x, y: 1 / ((x + 1.02)**2 + (y + 1.02)**2), tol=EP)
    _report(g)
    _pivot_plot(g, 2, [-1, 0.2, 1], [-1, 0.5, 1])


if __name__ == "__main__":
    run()
