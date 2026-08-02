"""Does a chebfun of degree n have n roots?

Faithful replica of roots/FundamentalTheoremOfAlgebra.m by Alex
Townsend (October 2011): counting the roots of chebfuns in the
complex plane via roots(f, 'all'), with Chebfun-ellipse plots.

Original: https://www.chebfun.org/examples/roots/FundamentalTheoremOfAlgebra.html
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
from chebfunjax.plotting import chebfun_style, plotregion

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _root_plot(f, r, seg, title):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.plot(seg, [0, 0], 'k-', lw=3)
    r = np.asarray(r)
    ax.plot(r.real, r.imag, '.r', ms=10)
    plotregion(f, ax=ax, title="")
    ax.set_xlabel("Re", fontsize=14)
    ax.set_ylabel("Im", fontsize=14)
    ax.legend(["interval", "Computed roots", "Chebfun ellipse"],
              loc="upper right")
    ax.set_title(title, fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"FundamentalTheoremOfAlgebra_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    n = 100
    rs = np.random.RandomState(5489)
    f = cj.Chebfun.from_values(jnp.asarray(rs.rand(n + 1)))
    r = np.asarray(f.roots(all_roots=True))
    print(f"This chebfun of degree {len(f)-1} has {len(r)} roots")

    f = cj.chebfun(lambda x: jnp.exp(-10 * x))
    r = np.asarray(f.roots(all_roots=True))
    _root_plot(f, r, [-1, 1],
               f"Degree {len(f)-1} with {len(r)} roots")

    n = 71
    xx = np.arange(0, n + 1) / n

    def wilkinson(x):
        return jnp.prod(jnp.asarray(x)[..., None] - xx, axis=-1)

    f = cj.chebfun(wilkinson, domain=(0.0, 1.0))
    r = np.asarray(f.roots())
    _root_plot(f, r + 0j, [0, 1],
               f"Degree {len(f)-1} with {len(r)} roots")

    rreal = np.asarray(f.roots())
    rall = np.asarray(f.roots(all_roots=True))
    print(f"No. of real roots = {len(rreal)}")
    print(f"No. of complex (and real) roots = {len(rall)}")
    print("ans =")
    print(f"     {np.linalg.norm(np.asarray(f(rreal))):.15e}")


if __name__ == "__main__":
    run()
