"""Eight Shades of Rational Approximation.

Translation of approx/EightShades.m, Mohsin Javed and Nick Trefethen,
January 2016. Original: https://www.chebfun.org/examples/approx/EightShades.html
Copyright by The University of Oxford and The Chebfun Developers.
Numerical source: Chebfun 7574c77680d7e82b79626300bf255498271a72df.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import chebfun_style, matlab_plot, save_chebfun_figure

_HERE = Path(__file__).resolve().parent
_IMG = _HERE.parents[1] / "docs" / "images" / "approx"
_CAPTURE = os.environ.get("CHEBFUNJAX_CAPTURE_DIR")
_YL = (-0.5, 1.2)
_LABELS = ("interpolation", "projection", "minimax", "CF")


def _capture(name, f):
    if not _CAPTURE:
        return
    directory = Path(_CAPTURE)
    directory.mkdir(parents=True, exist_ok=True)
    native_x = os.environ.get("CHEBFUNJAX_NATIVE_SAMPLE_X")
    x = np.load(native_x) if native_x else np.linspace(-1, 1, 501)
    fields = {
        "sample_x": x,
        "sample_values": np.asarray(f(jnp.asarray(x))),
        "point_values": np.asarray(f.point_values),
        "domain": np.asarray(f.domain.breakpoints),
    }
    for i, piece in enumerate(f.funs):
        fields[f"coefficients_{i}"] = np.asarray(piece.coeffs)
        fields[f"interval_{i}"] = np.asarray(piece.interval)
    np.savez(directory / f"{name}.npz", **fields)


def _stage(name):
    # Stage markers identify actual warning ownership in retained stdout/stderr.
    if _CAPTURE:
        print(f"SOURCE_STAGE {name}", flush=True)


def _figure(index, f=None, approximants=(), color="r"):
    fig, axes = plt.subplots(2, 2, figsize=(600 / 72, 270 / 72), dpi=72)
    # Observed R2024b CPU Painters subplot positions at 600 by 270 pixels.
    # Figure size precedes subplot construction; ylim follows the source plot.
    positions = (
        (0.13, 0.5838372093023255, 0.3346590909090909, 0.3411627906976745),
        (0.5703409090909091, 0.5838372093023255, 0.3346590909090909, 0.3411627906976745),
        (0.13, 0.11, 0.3346590909090909, 0.3411627906976744),
        (0.5703409090909091, 0.11, 0.3346590909090909, 0.3411627906976744),
    )
    metadata = []
    for i, (ax, label) in enumerate(zip(axes.flat, _LABELS, strict=True)):
        ax.set_position(positions[i])
        # Captured R2024b source axes: width .5pt, inward ticks, Box on.
        # TickLength .01 is normalized to the longest visible axes line.
        tick_length = 0.01 * max(600 * positions[i][2], 270 * positions[i][3])
        ax.tick_params(
            labelsize=10, direction="in", colors=(0.15, 0.15, 0.15),
            width=0.5, length=tick_length, top=True, right=True,
        )
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color((0.15, 0.15, 0.15))
        if i < len(approximants):
            matlab_plot(f, "k", approximants[i], color, ax=ax)
            ax.set_xlim(-1, 1)
            ax.set_ylim(*_YL)
            ax.set_xticks((-1, -0.5, 0, 0.5, 1))
            ax.set_yticks((-0.5, 0, 0.5, 1))
        else:
            ax.text(-0.5, 0.2, "(not yet available)", fontsize=10)
            ax.set_xlim(-1, 1)
            ax.set_ylim(*_YL)
            ax.set_xticks([])
            ax.set_yticks([])
            # Native Box off keeps the bottom/left axes, without ticks.
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax.text(-0.93, 0.9, label, fontsize=10)
        line_properties = []
        for line in ax.lines:
            line.set_linewidth(0.5)
            line_properties.append(
                {
                    "color": line.get_color(),
                    "linewidth": line.get_linewidth(),
                    "linestyle": line.get_linestyle(),
                    "marker": line.get_marker(),
                    "markersize": line.get_markersize(),
                    "markeredgewidth": line.get_markeredgewidth(),
                }
            )
        metadata.append(
            {
                "position": list(ax.get_position().bounds),
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim()),
                "xticks": ax.get_xticks().tolist(),
                "yticks": ax.get_yticks().tolist(),
                "lines": line_properties,
                "frame_on": ax.get_frame_on(),
            }
        )
        if _CAPTURE:
            for k, line in enumerate(ax.lines):
                np.savez(
                    Path(_CAPTURE) / f"figure_{index}_axis_{i}_line_{k}.npz",
                    x=np.asarray(line.get_xdata()),
                    y=np.asarray(line.get_ydata()),
                )
    _IMG.mkdir(parents=True, exist_ok=True)
    save_chebfun_figure(fig, _IMG / f"EightShades_{index:02d}.png", size=(600, 270), dpi=72)
    if _CAPTURE:
        (Path(_CAPTURE) / f"figure_{index}_axes.json").write_text(json.dumps(metadata, indent=2))
    plt.close(fig)


def run():
    chebfun_style()
    _stage("f")
    f = cj.chebfun(lambda x: jnp.exp(-50 * (x - 0.1) ** 2), trig=True)
    _capture("f", f)
    m = 8
    _stage("p1")
    p1 = cj.chebfun(f, n=m + 1)
    _stage("p2")
    p2 = cj.chebfun(f, trunc=m + 1)
    _stage("p3")
    p3 = cj.remez(f, m)
    _stage("p4")
    p4 = cj.cf(f, m)[0]
    for name, value in (("p1", p1), ("p2", p2), ("p3", p3), ("p4", p4)):
        _capture(name, value)
    _figure(1, f, (p1, p2, p3, p4))
    _stage("CFerror")
    CFerror = (p3 - p4).norm(np.inf)
    print("CFerror =")
    print(f"     {float(CFerror):.4e}")
    _stage("t1")
    t1 = cj.chebfun(f, n=m + 1, trig=True)
    _stage("t2")
    t2 = cj.chebfun(f, trunc=m + 1, trig=True)
    _stage("t3")
    t3 = cj.trigremez(f, m // 2)[0]
    for name, value in (("t1", t1), ("t2", t2), ("t3", t3)):
        _capture(name, value)
    _figure(2, f, (t1, t2, t3), "b")
    m = n = 3
    _stage("r1")
    # Existing Python ratinterp returns coefficients rather than native p,q.
    # Reconstruct those Chebyshev polynomials before literal source p/q.
    _, pc, qc, *_ = cj.ratinterp(f, m, n)
    p = Chebfun.from_coeffs(pc, f.domain)
    q = Chebfun.from_coeffs(qc, f.domain)
    _capture("r1_p", p)
    _capture("r1_q", q)
    r1 = p / q
    _stage("r2")
    p, q, _ = cj.chebpade(f, m, n)
    _capture("r2_p", p)
    _capture("r2_q", q)
    r2 = p / q
    _stage("r3")
    p, q = cj.remez(f, m, n)
    _capture("r3_p", p)
    _capture("r3_q", q)
    r3 = p / q
    _stage("r4")
    p, q, *_ = cj.cf(f, m, n)
    _capture("r4_p", p)
    _capture("r4_q", q)
    r4 = p / q
    for name, value in (("r1", r1), ("r2", r2), ("r3", r3), ("r4", r4)):
        _capture(name, value)
    _figure(3, f, (r1, r2, r3, r4))
    _figure(4)


if __name__ == "__main__":
    run()
