"""Low-rank approximation and alignment with axes (Chebfun2).

Demonstrates how axis-alignment affects the numerical rank of 2D functions,
following the Chebfun example approx2/Alignment.m by Nick Trefethen (April 2016).

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.plotting import surf, contour

def run():
    print("=" * 60)
    print("Low-rank approximation and alignment with axes")
    print("=" * 60)

    # tanh(k*x) depends only on x — rank 1
    k = 3
    f = cj.chebfun2(lambda x, y: jnp.tanh(k * x))
    print(f"\ntanh(3x) rank: {f.rank}  (expected 1)")
    assert f.rank == 1

    # tanh(k*(x+y)) mixes x and y — higher rank
    g = cj.chebfun2(lambda x, y: jnp.tanh(k * (x + y)))
    print(f"tanh(3(x+y)) rank: {g.rank}  (expected >> 1)")
    assert g.rank > 1

    # Rotated Gaussian: rotation kills separability
    h = cj.chebfun2(lambda x, y: jnp.exp(-((x + y)**2 + (x - y)**2) / 2))
    print(f"exp(-(x+y)^2/2 - (x-y)^2/2) rank: {h.rank}")

    # Plot
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    xs = np.linspace(-1, 1, 60)
    X, Y = np.meshgrid(xs, xs)
    Z1 = np.tanh(k * X)
    Z2 = np.tanh(k * (X + Y))

    axes[0].contourf(X, Y, Z1, levels=20, cmap="RdBu_r")
    axes[0].set_title(f"tanh(3x): rank {f.rank}", fontsize=12)

    axes[1].contourf(X, Y, Z2, levels=20, cmap="RdBu_r")
    axes[1].set_title(f"tanh(3(x+y)): rank {g.rank}", fontsize=12)

    fig.suptitle("Rank vs. axis alignment", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "low_rank_alignment.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
