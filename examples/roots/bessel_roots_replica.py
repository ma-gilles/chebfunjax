"""Roots of a Bessel function.

Faithful replica of roots/BesselRoots.m by Nick Trefethen
(September 2010, revised June 2019): all roots of J0 on [0,100] from
a single roots() call, and root counting on distant intervals.

Original: https://www.chebfun.org/examples/roots/BesselRoots.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')


def besselj0(x):
    return jnp.asarray(sp.j0(np.asarray(x)))


def run():
    os.makedirs(_IMG, exist_ok=True)

    J0 = cj.chebfun(besselj0, domain=(0.0, 100.0))
    xs = np.linspace(0, 100, 2000)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, np.asarray(J0(xs)), 'b', lw=1.2)
    ax.grid(True)
    ax.set_title("Bessel function J_0", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "BesselRoots_repl_01.png"),
                dpi=150, bbox_inches="tight")

    r = np.asarray(J0.roots())
    ax.plot(r, np.asarray(J0(r)), '.r', ms=9)
    fig.savefig(os.path.join(_IMG, "BesselRoots_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("number_of_roots =")
    print(f"    {len(r)}")

    def rootsab(a, b):
        f = cj.chebfun(besselj0, domain=(float(a), float(b)))
        return len(np.asarray(f.roots()))

    t0 = time.time()
    print("Number of roots between 1000000 and 1001000:")
    n = rootsab(1000000, 1001000)
    print("n =")
    print(f"   {n}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
