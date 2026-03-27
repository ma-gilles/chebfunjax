"""Random functions in 2D (Chebfun2).

Demonstrates constructing random smooth 2D functions and exploring their
statistical properties, following approx2/Random2D.m by Nick Trefethen (April 2017).

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.plotting import surf


def random_chebfun2(key, n=8):
    """Build a random smooth 2D function from random Chebyshev coefficients."""
    k1, k2 = jax.random.split(key)
    # Random low-rank: sum of rank-1 terms exp(-x^2)*exp(-y^2) type
    xs = jax.random.normal(k1, (n,)) * 0.5
    ys = jax.random.normal(k2, (n,)) * 0.5

    def f(x, y):
        val = jnp.zeros_like(x + y)
        for i in range(n):
            val = val + jnp.exp(-4 * (x - xs[i])**2) * jnp.exp(-4 * (y - ys[i])**2)
        return val

    return f


def run():
    print("=" * 60)
    print("Random functions in 2D")
    print("=" * 60)

    key = jax.random.PRNGKey(42)

    # Build several random 2D functions
    fns = []
    for i in range(4):
        key, subkey = jax.random.split(key)
        fn = random_chebfun2(subkey)
        f = cj.chebfun2(fn)
        fns.append(f)
        print(f"Random function {i+1}: rank {f.rank}")

    # Verify evaluations are finite
    for i, f in enumerate(fns):
        val = float(f(jnp.array(0.0), jnp.array(0.0)))
        assert np.isfinite(val), f"Function {i} evaluation not finite"

    # Plot
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    xs = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(xs, xs)
    key = jax.random.PRNGKey(42)
    for i, ax in enumerate(axes):
        key, subkey = jax.random.split(key)
        fn = random_chebfun2(subkey)
        Z = np.array(fn(jnp.array(X), jnp.array(Y)))
        im = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
        ax.set_title(f"Random 2D function {i+1}", fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Random smooth 2D functions", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "random_functions_2d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
