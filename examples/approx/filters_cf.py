"""Digital filters via CF approximation.

Translation of approx/FiltersCF.m by Nick Trefethen (September
2014): high-degree polynomial CF approximations of a square-wave
filter shape, and of its mollified (triangular-kernel) smoothing.

Original: https://www.chebfun.org/examples/approx/FiltersCF.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.cfpade import cf

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

FIG = [0]


def _plot(*args, axis=(-1, 1, -1, 2)):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    matlab_plot(*args, ax=ax, linewidth=1.2,
                jumpline={'linestyle': ':', 'color': 'k'})
    if axis is not None:
        ax.axis(axis)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"FiltersCF_{FIG[0]:02d}.png"),
             size=(598, 273))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    x = cj.chebfun('x')
    f = (abs(x) < .3) + (abs(x - .7) < .1) + (abs(x + .65) < .2)
    # plot(f,'k',...), shg: overwritten before the first snapnow, so the
    # published page shows no figure for it.

    t0 = time.time()
    for m in (100, 1000):
        p, q, rh, s = cf(f, m, 0, max(100, 2 * m))
        _plot(f, 'k', p, 'r')
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    s = cj.chebfun('s', domain=[-.02, .02])
    phi = 50 - 50**2 * abs(s)
    f2 = f.conv(phi)
    for m in (100, 200):
        p, q, rh, s = cf(f2, m, 0, max(100, 2 * m))
        _plot(f2, 'k', p, 'r')

    # plot(f2-p), grid on: default axis limits
    _plot(f2 - p, axis=None)


if __name__ == "__main__":
    run()
