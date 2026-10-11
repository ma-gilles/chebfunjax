"""Field of values.

Translation of linalg/FieldOfValues.m by Nick Trefethen
(November 2010): the field of values (numerical range) of a matrix as
a chebfun of the boundary parametrized by angle, computed by
Johnson's algorithm; the numerical abscissa; and the polygonal /
line-segment cases of normal and non-generic matrices.

The matrix is captured from MATLAB R2025b rng(1), randn(20), with
provenance beside the script. Numerical operations use the library.

Original: https://www.chebfun.org/examples/linalg/FieldOfValues.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import inspect
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax.chebfun1d.fov as fov_module
from chebfunjax.chebfun1d.fov import fov
from chebfunjax.plotting import _source_plot_line_and_points, chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

FIG = [0]


def _num(v):
    """MATLAB format-long display of a real scalar."""
    return f"{int(v):6d}" if float(v) == int(v) else f"   {v:.15f}"


def _save(fig, close=True):
    FIG[0] += 1
    # Observed native source geometry, R2024b CPU/Painters at 600x270.
    # Numerical plot coordinates are unchanged by this rendering adapter.
    native = json.loads(Path(__file__).with_name(
        "field_of_values_native_axes.json").read_text())["figures"][str(FIG[0])]
    ax = fig.axes[0]
    ax.set_position(native["position"])
    ax.axis(native["limits"])
    ax.set_xticks(native["xticks"])
    ax.set_yticks(native["yticks"])
    ax.set_aspect("equal", adjustable="box")
    fig.set_facecolor("white")
    fig.canvas.draw()
    capture = os.environ.get("CHEBFUNJAX_CAPTURE_DIR")
    if capture:
        dest = Path(capture)
        dest.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for ai, ax in enumerate(fig.axes):
            arrays[f"axes_{ai}_limits"] = np.asarray(ax.axis())
            for li, line in enumerate(ax.lines):
                arrays[f"axes_{ai}_line_{li}"] = line.get_xydata()
        np.savez(dest / f"FieldOfValues_{FIG[0]:02d}.npz", **arrays)
    _savefig(fig, os.path.join(
        _IMG, f"FieldOfValues_{FIG[0]:02d}.png"), dpi=72)
    if close:
        plt.close(fig)


def _dot(ax, x, y, color, size):
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    _source_plot_line_and_points(ax, x, y, x, y, '.' + color,
                                 {'markersize': size})


def _plot_fov(F, eigs, axis_lim=None):
    fig, ax = plt.subplots(figsize=(600 / 72, 270 / 72), dpi=72)
    ax.set_position((.13, .11, .775, .815))
    bps = [float(v) for v in F.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 300)
        v = np.asarray(F(t))
        ax.plot(v.real, v.imag, 'b', lw=1.6)
    # 'jumpline' {'b'}: join the one-sided values at each interior break
    for bp in bps[1:-1]:
        zl = complex(np.asarray(F(bp, 'left')))
        zr = complex(np.asarray(F(bp, 'right')))
        ax.plot([zl.real, zr.real], [zl.imag, zr.imag], 'b', lw=1.6)
    _dot(ax, eigs.real, eigs.imag, 'k', 18)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    if axis_lim:
        ax.axis(axis_lim)
    else:
        fig.canvas.draw()
        ax.axis(1.1 * np.asarray(ax.axis()))  # source axis(1.1*ax)
    return fig, ax


def run():
    os.makedirs(_IMG, exist_ok=True)

    fixture = json.loads(Path(__file__).with_name("field_of_values_matlab_rng1.json").read_text())
    A = np.asarray(fixture["A"])
    FA = fov(A)
    eigsA = np.linalg.eigvals(A)
    fig, ax = _plot_fov(FA, eigsA)
    reference_limits = ax.axis()
    _save(fig, close=False)

    reF = FA.real()
    maxtheta, alpha = reF.max()
    print("@@CHEBFUNJAX_OUTPUT_SLOT:1@@")
    print("alpha =")
    print(f"   {float(alpha):.15f}")
    print("maxtheta =")
    print(_num(float(maxtheta)))
    zmax = complex(np.asarray(FA(float(maxtheta))))
    _dot(ax, zmax.real, zmax.imag, 'r', 24)
    _save(fig)

    print("@@CHEBFUNJAX_OUTPUT_SLOT:2@@")
    alpha2 = np.max(np.linalg.eigvalsh((A + A.T) / 2))
    print("alpha =")
    print(f"   {alpha2:.15f}")

    B = np.diag(eigsA)
    FB = fov(B)
    fig, ax = _plot_fov(FB, eigsA, axis_lim=reference_limits)
    reB = FB.real()
    mth, _ = reB.max()
    zb = complex(np.asarray(FB(float(mth))))
    _dot(ax, zb.real, zb.imag, 'r', 24)
    _save(fig)
    print("@@CHEBFUNJAX_OUTPUT_SLOT:3@@")
    print("FB =")
    print(repr(FB))

    C = np.array([[0, 3, 0, 0],
                  [-3, 0, 0, 0],
                  [0, 0, 0, 3],
                  [0, 0, 1, 1]], dtype=float)
    print("@@CHEBFUNJAX_OUTPUT_SLOT:4@@")
    print("C =")
    for row in C:
        print("".join(f"{int(v):6d}" for v in row))
    FC = fov(C)
    eigsC = np.linalg.eigvals(C)
    fig, ax = _plot_fov(FC, eigsC, axis_lim=[-4, 4, -4, 4])
    _save(fig)

    print("@@CHEBFUNJAX_OUTPUT_SLOT:5@@")
    print(inspect.getsource(fov_module))
    print("@@CHEBFUNJAX_OUTPUT_SLOT:0@@")


if __name__ == "__main__":
    run()
