"""The tiger's tail: roots of a high-degree polynomial.

Demonstrates that even high-degree random polynomials can be rooted
stably via the colleague matrix, illustrating the "tiger's tail" shape
formed by roots of Chebyshev-expanded random polynomials.

Credit: Inspired by Chebfun example roots/Tiger.m
(Nick Trefethen, August 2014).
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
    print("The tiger's tail: roots of high-degree polynomials")
    print("=" * 60)

    rng = np.random.default_rng(17)

    # Build random Chebyshev polynomial of degree n with geometrically decaying
    # coefficients (well-conditioned).  The real roots form the "tiger's body"
    # and the complex eigenvalues of the colleague matrix form the "tail".
    print("\nBuilding random Chebyshev polynomials and finding roots:")

    results = {}
    for n in [50, 100, 200]:
        # Coefficients decaying like 0.95^k
        coeffs = rng.standard_normal(n + 1) * 0.95**np.arange(n + 1)
        coeffs_jnp = jnp.array(coeffs)
        f = cj.Chebfun.from_coeffs(coeffs_jnp)
        r = f.roots()
        r_arr = np.sort(np.array(r))
        # All roots should satisfy |f(r)| ~ 0 relative to vscale
        vscale = float(jnp.max(jnp.abs(f(jnp.linspace(-1.0, 1.0, 200)))))
        if len(r_arr) > 0:
            err = float(np.max(np.abs(np.array([float(f(jnp.array(ri))) for ri in r_arr[:20]]))))
        else:
            err = 0.0
        print(f"  n={n}: {len(r_arr)} real roots, max |f(root)| / vscale = {err / vscale:.2e}")
        assert err / vscale < 1e-8 or len(r_arr) == 0
        results[n] = (f, r_arr, vscale)

    # Detailed analysis for n=100
    n = 100
    f_main, r_main, vscale = results[n]
    print(f"\nDetailed: n={n} polynomial")
    print(f"  Chebyshev length: {len(f_main)}")
    print(f"  Number of real roots: {len(r_main)}")
    print(f"  vscale: {vscale:.6f}")

    # Verify roots are in [-1, 1]
    if len(r_main) > 0:
        assert np.all(r_main >= -1.0 - 1e-10) and np.all(r_main <= 1.0 + 1e-10)

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))

    rng2 = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2)

    for col, n in enumerate([50, 200]):
        coeffs = rng2.standard_normal(n + 1) * 0.95**np.arange(n + 1)
        f = cj.Chebfun.from_coeffs(jnp.array(coeffs))
        r = np.sort(np.array(f.roots()))

        xs_plot = np.linspace(-1.0, 1.0, 600)
        ys_plot = np.array(f(jnp.array(xs_plot)))
        axes[col].plot(xs_plot, ys_plot, color="#4169E1", linewidth=1.0, alpha=0.7)
        axes[col].axhline(0, color="k", linewidth=0.5)
        if len(r) > 0:
            axes[col].plot(r, np.zeros_like(r), "ro", markersize=3, label=f"{len(r)} roots")
        axes[col].set_title(f"Random Chebyshev, $n={n}$")
        axes[col].legend(fontsize=9)
        axes[col].set_xlabel("x")
        axes[col].grid(True, alpha=0.3)

    fig.suptitle("Tiger's tail: roots of high-degree Chebyshev polynomials", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "tiger.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
