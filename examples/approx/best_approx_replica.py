"""Polynomial and rational best approximation of |x-0.5|.

Faithful replica of approx/BestApprox.m by Nick Trefethen (October
2010): equioscillating error curves for the degree-16 polynomial and
type (8,8)/(16,16) rational minimax approximations to |x-1/2|, with
zooms showing the error concentrated near the singularity.

Original: https://www.chebfun.org/examples/approx/BestApprox.html
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
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def _errplot(fh, rh, err, dom, ylim, title, fname):
    xs = np.linspace(dom[0], dom[1], 3000)
    ev = np.asarray(fh(jnp.asarray(xs))) - np.asarray(rh(xs))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(xs, ev, lw=1.6)
    ax.plot([dom[0], dom[1]], [err, err], '--k', lw=1)
    ax.plot([dom[0], dom[1]], [-err, -err], '--k', lw=1)
    ax.set_xlim(*dom)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    f = lambda x: jnp.abs(x - 0.5)  # noqa: E731

    # Degree 16 polynomial minimax
    res = minimax(f, 16)
    print("poly err =")
    print(f"   {res.err:.15f}")
    p_cf = cj.chebfun(jnp.asarray(res.coeffs), coeffs=True)
    _errplot(f, lambda x: np.asarray(p_cf(jnp.asarray(x))), res.err,
             (-1, 1), (-0.03, 0.03),
             "Degree 16 polynomial error curve",
             "BestApprox_repl_01.png")

    # Type (8,8) rational minimax
    r88 = minimax(f, 8, rational=True, denom=8)
    print("rat88 err =")
    print(f"   {r88.err:.15e}")
    _errplot(f, r88.r, r88.err, (-1, 1), (-0.003, 0.003),
             "Type (8,8) rational error curve",
             "BestApprox_repl_02.png")

    # Type (16,16) rational minimax
    r16 = minimax(f, 16, rational=True, denom=16)
    print("rat1616 err =")
    print(f"   {r16.err:.15e}")
    _errplot(f, r16.r, r16.err, (-1, 1), (-4e-5, 4e-5),
             "Type (16,16) rational error curve",
             "BestApprox_repl_03.png")

    # Zooms near the singularity
    _errplot(f, r16.r, r16.err, (0.45, 0.55), (-4e-5, 4e-5),
             "Zoom near singularity", "BestApprox_repl_04.png")
    _errplot(f, r16.r, r16.err, (0.4975, 0.5025), (-4e-5, 4e-5),
             "Closer zoom", "BestApprox_repl_05.png")


if __name__ == "__main__":
    run()
