"""Pretty functions approximated by Chebfun2.

A gallery of visually striking 2D functions approximated by chebfunjax,
following approx2/PrettyFunctions.m by Alex Townsend (March 2013).

Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj


FUNCTIONS = [
    ("sin(10x) cos(10y)", lambda x, y: jnp.sin(10 * x) * jnp.cos(10 * y)),
    ("exp(-x²-y²)", lambda x, y: jnp.exp(-x**2 - y**2)),
    ("sin(x²+y²)", lambda x, y: jnp.sin(x**2 + y**2)),
    ("tanh(5(x²-y²))", lambda x, y: jnp.tanh(5 * (x**2 - y**2))),
]


def run():
    print("=" * 60)
    print("Pretty functions approximated by Chebfun2")
    print("=" * 60)

    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    xs = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(xs, xs)

    for i, (title, fn) in enumerate(FUNCTIONS):
        f = cj.chebfun2(fn)
        print(f"{title}: rank {f.rank}")

        # Verify evaluation
        v = float(f(jnp.array(0.0), jnp.array(0.0)))
        v_exact = float(fn(jnp.array(0.0), jnp.array(0.0)))
        assert abs(v - v_exact) < 1e-8, f"Mismatch for {title}"

        Z = np.array(fn(jnp.array(X), jnp.array(Y)))
        im = axes[i].contourf(X, Y, Z, levels=30, cmap="viridis")
        axes[i].set_title(f"{title}\nrank {f.rank}", fontsize=10)
        axes[i].set_xlabel("x"); axes[i].set_ylabel("y")
        fig.colorbar(im, ax=axes[i], shrink=0.8)

    fig.suptitle("Pretty 2D functions in chebfunjax", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "pretty_functions_2d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
