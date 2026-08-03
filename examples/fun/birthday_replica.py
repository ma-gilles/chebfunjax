"""Happy Birthday Pafnuty!

Faithful replica of fun/Birthday.m by Nick Trefethen (May 2011,
Chebyshev's 190th birthday): a scribbled greeting mapped through
analytic functions of a complex variable.

Original: https://www.chebfun.org/examples/fun/Birthday.html
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
from chebfunjax.utils.scribble import scribble

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fun')

FIG = [0]


def _plot_mapped(cf, fn, color, axis_off=False):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    bps = [float(v) for v in cf.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 30)
        z = fn(np.asarray(cf(t)))
        ax.plot(z.real, z.imag, color, lw=1.8)
    ax.set_aspect("equal")
    if axis_off:
        ax.set_axis_off()
    else:
        ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Birthday_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    s = scribble("Happy Birthday Pafnuty!")
    print("s =")
    print(repr(s))
    _plot_mapped(s, lambda z: z, 'b')
    _plot_mapped(s, np.exp, 'b')
    _plot_mapped(s, lambda z: np.exp(3j * z), 'm')
    _plot_mapped(s, lambda z: np.exp((1 + 1j) * z), 'g',
                 axis_off=True)
    _plot_mapped(s, lambda z: np.sinh(3 * z), 'r', axis_off=True)


if __name__ == "__main__":
    run()
