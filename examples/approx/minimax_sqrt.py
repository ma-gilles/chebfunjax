"""Approximating square and fifth roots by polynomials and rational functions.

Translation of approx/MinimaxSqrt.m by Yuji Nakatsukasa, May 2019.
Original: https://www.chebfun.org/examples/approx/MinimaxSqrt.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, save_chebfun_figure
from chebfunjax.utils.minimax import minimax

_HERE = Path(__file__).resolve().parent
_IMG = Path(os.environ.get("MINIMAXSQRT_IMAGE_DIR", _HERE.parent.parent / "docs/images/approx"))


def _sweep(f, a, ns, *, silent=False, separate=False):
    """Execute the native loop order, preserving solver failures and warnings."""
    p, r = [], []

    def polynomial(n):
        return minimax(f, int(n), domain=(a, 1.)).err

    def rational(n):
        return minimax(f, int(n)//2, rational=True, denom=int(n)//2,
                       domain=(a, 1.), silent=silent).err

    if separate:
        p = [polynomial(n) for n in ns]
        r = [rational(n) for n in ns]
    else:
        for n in ns:
            p.append(polynomial(n))
            r.append(rational(n))
    return np.asarray(p), np.asarray(r)


def _draw(ax, a, ns, errors, *, pad, styles=("b*-", "r*-"), label_factor=1.):
    perrs, rerrs = errors
    ax.semilogy(ns, perrs, styles[0])
    ax.text(ns[-1]+pad, perrs[-1]*label_factor, f"poly a={a:g}", color="b")
    ax.semilogy(ns, rerrs, styles[1])
    ax.text(ns[-1]+pad, rerrs[-1], f"rat a={a:g}", color="r")
    ax.grid(True)
    ax.set_xlim(0, ns[-1]+(2 if ns[-1] == 8 else 7))
    ax.set_xlabel("DOF")
    ax.set_title(f"sqrt(x) on [a,1], a = {a:g}")


def _save(fig, number):
    save_chebfun_figure(fig, _IMG/f"MinimaxSqrt_{number:02d}.png", size=(600, 270), layout="matlab")


def run():
    """Execute all seven literal approximation sweeps and six source plots.

    Provenance
    ----------
    Chebfun example approx/MinimaxSqrt.m, Yuji Nakatsukasa, May 2019.
    MATLAB reference library Chebfun commit: 7574c77.
    """
    chebfun_style()
    _IMG.mkdir(parents=True, exist_ok=True)
    captures = {}
    for i, a in enumerate((.8, .1, 1e-3, 1e-5, 0.), 1):
        ns = np.arange(2, 9 if i <= 2 else 21, 2)
        if i <= 2:
            x = cj.chebfun("x", domain=(a, 1.))
            f = cj.sqrt(x)
        else:
            f = cj.chebfun(lambda x: jnp.sqrt(x), domain=(a, 1.), splitting=i >= 4)
        errors = _sweep(f, a, ns, silent=i >= 3)
        captures[f"sqrt_{i}_poly"], captures[f"sqrt_{i}_rat"] = errors
        if i != 5:
            fig, ax = plt.subplots()
        _draw(ax, a, ns, errors, pad=.25 if i <= 2 else .5,
              styles=("bo--", "ro--") if i == 5 else ("b*-", "r*-"),
              label_factor=1.3 if i == 5 else 1.)
        _save(fig, i)
        if i != 4:
            plt.close(fig)
    fig, ax = plt.subplots()
    for i, a in enumerate((1e-5, 0.)):
        f = cj.chebfun(lambda x: x**(1/5), domain=(a, 1.), splitting=True)
        errors = _sweep(f, a, ns, silent=True, separate=True)
        captures[f"fifth_{i}_poly"], captures[f"fifth_{i}_rat"] = errors
        _draw(ax, a, ns, errors, pad=.2,
              styles=("b*-", "r*-") if i == 0 else ("bo--", "ro--"))
    ax.set_title("fifth root of x on [a,1], a = 0 and 1e-5")
    _save(fig, 6)
    plt.close(fig)
    if directory := os.environ.get("MINIMAXSQRT_CAPTURE_DIR"):
        Path(directory).mkdir(parents=True, exist_ok=True)
        np.savez(Path(directory)/"errors.npz", **captures)


if __name__ == "__main__":
    run()
