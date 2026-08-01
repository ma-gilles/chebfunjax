"""Greedy interpolation.

Faithful replica of approx/GreedyInterp.m by Nick Trefethen (October
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

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.lebesgue import lebesgue_function
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 20001)


def run():
    os.makedirs(_IMG, exist_ok=True)

    fv = np.abs(XS)
    s = []
    maxpos = 0.0     # norm(|x|, inf) attained at x = +-1; MATLAB gets +-1
    maxval = 1.0
    maxpos = 1.0
    fignum = 0
    for n in range(0, 129):
        s.append(maxpos)
        p = Chebfun.interp1(jnp.asarray(np.asarray(s, dtype=np.float64)),
                            jnp.asarray(np.abs(np.asarray(s))),
                            domain=(-1.0, 1.0))
        ev = fv - np.asarray(p(jnp.asarray(XS)))
        i = int(np.argmax(np.abs(ev)))
        maxval = abs(ev[i])
        maxpos = XS[i]
        if n <= 4 or (n >= 5 and np.log2(n) == round(np.log2(n))):
            fignum += 1
            fig, ax = plt.subplots(figsize=(8.8, 3.6))
            ax.plot(XS, ev, lw=2)
            ax.set_ylim(-1.2 * maxval, 1.2 * maxval)
            ax.grid(True)
            ax.plot([maxpos], [ev[i]], '.r', ms=18)
            ax.set_title(f"n = {n}    error = {maxval:.5g}", fontsize=13)
            fig.set_facecolor("white")
            fig.tight_layout()
            fig.savefig(os.path.join(
                _IMG, f"GreedyInterp_repl_{fignum:02d}.png"),
                dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"n = {n}   error = {maxval:.6g}")

    s = np.asarray(s)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(np.sort(s), '.k', ms=8)
    scheb = np.asarray(chebpts(len(s)))
    ax.plot(scheb, 'or', ms=4, mfc='none')
    ax.set_ylim(-1.02, 1.02)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"GreedyInterp_repl_{fignum+1:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    t1, l1 = lebesgue_function(s, n_eval=8001)
    t2, l2 = lebesgue_function(scheb, n_eval=8001)
    ax.semilogy(t1, l1, 'k', lw=1.4)
    ax.semilogy(t2, l2, 'r', lw=1.4)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"GreedyInterp_repl_{fignum+2:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"greedy Lebesgue max: {np.max(l1):.4g}")
    print(f"cheb   Lebesgue max: {np.max(l2):.4g}")


if __name__ == "__main__":
    run()
