"""Optimization over an integral.

Faithful replica of opt/OptimInt.m by Nick Trefethen
(October 2010): the function I(a) = integral of sin(x)+sin(a x^2),
studied as a chebfun of the parameter a — its level sets, maximum,
and the near-regular spacing of its local minima.

The inner integral has the closed form
2*sqrt(pi/(2a)) * FresnelS(sqrt(2a/pi)), used here so the outer
chebfun construction is exact and fast.

Original: https://www.chebfun.org/examples/opt/OptimInt.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import fresnel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'opt')


def I_of_a(a_arr):
    """I(a) = int_{-1}^{1} sin(x) + sin(a x^2) dx (exact Fresnel)."""
    a_arr = np.atleast_1d(np.asarray(a_arr, dtype=float))
    out = np.empty_like(a_arr)
    for i, a in enumerate(a_arr.ravel()):
        if a <= 0:
            out.ravel()[i] = 0.0
            continue
        s, _ = fresnel(np.sqrt(2 * a / np.pi))
        out.ravel()[i] = 2 * np.sqrt(np.pi / (2 * a)) * s
    return out.reshape(np.shape(a_arr))


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    Ia = cj.chebfun(lambda a: jnp.asarray(I_of_a(np.asarray(a))),
                    domain=(0.0, 100.0))
    r = np.asarray((Ia - 1).roots())
    print("r =")
    for v in r:
        print(f"   {v:.15f}")

    xs = np.linspace(0, 35, 900)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(Ia(xs)), lw=1.4)
    ax.grid(True)
    ax.axis([0, 35, 0, 1.2])
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.plot(r, np.asarray(Ia(r)), '.r', ms=10)
    r2 = np.asarray((Ia - 0.25).roots())
    print("r =")
    for v in r2:
        print(f"  {v:>18.15f}")
    ax.plot(r2, np.asarray(Ia(r2)), '.k', ms=10)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OptimInt_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    _, m = Ia.max()
    print("m =")
    print(f"   {float(m):.15f}")

    vals, poss = Ia.minandmax(flag="local")
    # local minima positions
    dIa = Ia.diff()
    crit = np.asarray(dIa.roots(nojump=True))
    d2 = dIa.diff()
    mins = crit[np.asarray(d2(crit)) > 0]
    spacing = np.diff(mins[1:-1])
    f = float(np.std(spacing, ddof=1))
    print("f =")
    print(f"   {f:.15f}")


if __name__ == "__main__":
    run()
