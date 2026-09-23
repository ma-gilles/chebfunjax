"""Greedy interpolation.

Translation of approx/GreedyInterp.m by Nick Trefethen (October
2011): interpolation nodes chosen greedily at the current error
maximum converge to a Chebyshev-like distribution.

Original: https://www.chebfun.org/examples/approx/GreedyInterp.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.lebesgue import lebesgue_function
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _norm_inf(g):
    """[maxval, maxpos] = norm(g, inf)."""
    (xmin, vmin), (xmax, vmax) = g.minandmax()
    if abs(float(vmin)) > abs(float(vmax)):
        return abs(float(vmin)), float(xmin)
    return abs(float(vmax)), float(xmax)


def _num2str(v):
    return f"{v:.{max(int(np.floor(np.log10(abs(v)))) + 5, 5)}g}"


def run():
    os.makedirs(_IMG, exist_ok=True)

    x = cj.chebfun(lambda t: t)
    f = cj.abs(x)

    s = []
    maxval, maxpos = _norm_inf(f)
    fignum = 0
    for n in range(0, 129):
        s.append(maxpos)
        sv = jnp.asarray(s)
        p = Chebfun.interp1(sv, f(sv), domain=(-1.0, 1.0))
        err = f - p
        maxval, maxpos = _norm_inf(err)
        if n <= 4 or np.log2(n) == round(np.log2(n)):
            fignum += 1
            fig, ax = plt.subplots(figsize=(6.0, 2.7))
            err.plot(ax=ax, linewidth=2, n_pts=4000)
            ax.set_ylim(-1.2 * maxval, 1.2 * maxval)
            ax.grid(True)
            ax.plot([maxpos], [float(err(maxpos))], '.r', ms=12)
            ax.set_title(f"n = {n}    error = {_num2str(maxval)}",
                         fontsize=10.5)
            fig.tight_layout()
            _savefig(fig, os.path.join(
                _IMG, f"GreedyInterp_{fignum:02d}.png"))
            plt.close(fig)

    # The greedy points (black) against Chebyshev points (red).
    s = np.asarray(s)
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.plot(np.arange(1, len(s) + 1), np.sort(s), '.k', ms=6)
    scheb = np.asarray(chebpts(len(s)))
    ax.plot(np.arange(1, len(s) + 1), scheb, 'or', ms=6, mfc='none')
    ax.set_ylim(-1.02, 1.02)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"GreedyInterp_{fignum + 1:02d}.png"))
    plt.close(fig)

    # Lebesgue functions of the greedy and Chebyshev points.
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    t1, l1 = lebesgue_function(s, n_eval=8001)
    t2, l2 = lebesgue_function(scheb, n_eval=8001)
    ax.semilogy(t1, l1, 'k', lw=1.0)
    ax.semilogy(t2, l2, 'r', lw=1.0)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"GreedyInterp_{fignum + 2:02d}.png"))
    plt.close(fig)


if __name__ == "__main__":
    run()
