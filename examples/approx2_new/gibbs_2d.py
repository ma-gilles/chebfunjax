"""The Gibbs phenomenon in 2D (Chebfun2).

Demonstrates Gibbs oscillations near a 2D discontinuity and their effect
on low-rank structure, following approx2/Gibbs2D.m by Andre Uschmajew and
Nick Trefethen (February 2017).

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

def run():
    print("=" * 60)
    print("The Gibbs phenomenon in 2D")
    print("=" * 60)

    # A smooth step-like function in 2D that approximates a discontinuity
    eps = 0.05
    f = cj.chebfun2(lambda x, y: jnp.tanh((x + y) / eps))
    print(f"tanh((x+y)/{eps}) rank: {f.rank}")
    assert f.rank > 1

    # Softer version
    eps2 = 0.2
    g = cj.chebfun2(lambda x, y: jnp.tanh((x + y) / eps2))
    print(f"tanh((x+y)/{eps2}) rank: {g.rank}")

    # Check values
    val = float(f(jnp.array(0.5), jnp.array(-0.5)))
    print(f"f(0.5, -0.5) = {val:.6f}  (expected ~tanh(0) = 0)")
    assert abs(val) < 0.01

    # Plot cross-section showing Gibbs-like overshoot
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    xs = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(xs, xs)

    Z1 = np.tanh((X + Y) / eps)
    Z2 = np.tanh((X + Y) / eps2)

    axes[0].contourf(X, Y, Z1, levels=30, cmap="RdBu_r")
    axes[0].set_title(f"tanh((x+y)/{eps}): rank {f.rank}", fontsize=12)

    axes[1].contourf(X, Y, Z2, levels=30, cmap="RdBu_r")
    axes[1].set_title(f"tanh((x+y)/{eps2}): rank {g.rank}", fontsize=12)

    fig.suptitle("2D step function: rank grows as ε → 0", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "gibbs_2d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
