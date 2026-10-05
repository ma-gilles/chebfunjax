"""Writing messages in 3D.

Translation of fun/Writing3D.m by Nick Trefethen
(November 2010): scribble text plotted flat, bent along a sine wave,
and wrapped around a cylinder in 3D.

Original: https://www.chebfun.org/examples/fun/Writing3D.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.scribble import scribble

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fun')

FIG = [0]


def _save(fig, size=(600, 270)):
    FIG[0] += 1
    fig.set_facecolor("white")
    if any(hasattr(ax, "zaxis") for ax in fig.axes):
        fig.tight_layout()
    else:
        fig.set_size_inches(size[0] / 100, size[1] / 100)
        for ax in fig.axes:
            # MATLAB default axes margins on the fixed published canvas.
            ax.set_position((0.13, 0.11, 0.775, 0.815))
    _savefig(fig, os.path.join(
        _IMG, f"Writing3D_{FIG[0]:02d}.png"), size=size)
    plt.close(fig)


def _pieces(cf, n=14):
    # Evaluate each stored Chebtech on its own reference interval. Evaluating
    # the full Chebfun at shared physical breakpoints averages jump values,
    # joining strokes that MATLAB's 'jumpline','none' keeps separate.
    # _Piece.interval is static, so sampling piece(x) would also specialize a
    # separate compiled function per physical interval. The canonical tech
    # uses [-1, 1] and preserves each piece's own endpoint trace.
    t = np.linspace(-1.0, 1.0, n)
    for piece in cf.funs:
        yield np.asarray(piece.tech(t))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    s = scribble("There is no fun like chebfun.")
    fig, ax = plt.subplots(figsize=(9.8, 2.6))
    for z in _pieces(s):
        ax.plot(z.real, z.imag, 'r', lw=2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig)

    # rs = real(s); is = imag(s); plot(rs,is,'m','jumpline','none')
    fig, ax = plt.subplots(figsize=(9.8, 2.6))
    for z in _pieces(s):
        ax.plot(z.real, z.imag, 'm', lw=2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig)

    fig = plt.figure(figsize=(9.8, 4.4))
    ax = fig.add_subplot(projection="3d")
    xyz = []
    for z in _pieces(s, 30):
        y = np.sin(6 * z.real)
        ax.plot(z.real, y, z.imag, color="b", lw=2, clip_on=False)
        xyz.append(np.stack((z.real, y, z.imag), axis=0))
    data = np.concatenate(xyz, axis=1)
    # MATLAB axis equal means one data unit has equal physical length on all
    # axes. Matplotlib expresses that with box dimensions proportional to the
    # plotted data ranges; the renderer's auto-limits may add margins.
    ranges = np.ptp(data, axis=1)
    ax.set_box_aspect(ranges, zoom=600 / 270)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    # MATLAB azimuth is measured from -y; Matplotlib's is from +x.
    ax.view_init(elev=6, azim=-91.5)
    _save(fig)

    s2 = 6 * scribble("There is no fun like chebfun.  "
                      "Try it and you'll see.  "
                      "It does your calculation, "
                      "and makes a cup of tea!")
    fig = plt.figure(figsize=(6.0, 4.5))
    ax = fig.add_subplot(projection="3d")
    for z in _pieces(s2, 30):
        rs = z.real
        ax.plot(np.cos(rs), np.sin(rs),
                z.imag + 0.05 * rs, color="#0072BD", lw=2)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_axis_off()
    # MATLAB -540 degrees maps to Matplotlib azimuth 90 degrees.
    ax.view_init(elev=20, azim=90)
    ax.set_box_aspect((1, 1, 1))
    _save(fig, size=(600, 450))


if __name__ == "__main__":
    run()
