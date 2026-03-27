"""Loosening the Chebfun3 tolerance for faster construction.

Demonstrates how using a looser tolerance speeds up Chebfun3 construction
while retaining acceptable accuracy for triple integrals. Machine precision
is the default but often not required.

Original MATLAB Chebfun: approx3/Tolerance.m by Nick Trefethen, June 2016.
See https://www.chebfun.org/examples/approx3/Tolerance.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.
"""

import matplotlib
matplotlib.use("Agg")
import os
import time

import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)

def run():
    print("=" * 60)
    print("Loosening Chebfun3 tolerance for speed (Tolerance)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Section 1: Machine-precision integral of simple function
    # I = int_{-1}^1^3 exp(sin(xyz + exp(xyz))) dz dy dx
    # ------------------------------------------------------------------
    print("\n--- Simple function: exp(sin(xyz + exp(xyz))) ---")

    t0 = time.time()
    f = chebfun3(lambda x, y, z: jnp.exp(jnp.sin(x * y * z + jnp.exp(x * y * z))))
    t_full = time.time() - t0
    I_full = float(f.sum3())
    print(f"  Machine precision (tol=eps):")
    print(f"    Tucker rank: {f.rank}")
    print(f"    I = {I_full:.15f}")
    print(f"    Time: {t_full:.3f}s")

    # ------------------------------------------------------------------
    # Section 2: More complex function at machine precision (slow)
    # g = exp(sin(10xyz + exp(xyz)))
    # ------------------------------------------------------------------
    print("\n--- Complex function: exp(sin(10xyz + exp(xyz))) ---")

    t0 = time.time()
    g_full = chebfun3(
        lambda x, y, z: jnp.exp(jnp.sin(10 * x * y * z + jnp.exp(x * y * z)))
    )
    t_g_full = time.time() - t0
    I_g_full = float(g_full.sum3())
    print(f"  Machine precision:")
    print(f"    Tucker rank: {g_full.rank}")
    print(f"    I = {I_g_full:.15f}")
    print(f"    Time: {t_g_full:.3f}s")

    # ------------------------------------------------------------------
    # Section 3: Looser tolerances — speedup comparison
    # ------------------------------------------------------------------
    print("\n--- Effect of tolerance on speed and accuracy ---")
    tol_vals = [1e-8, 1e-6, 1e-4]
    results = []

    for tol in tol_vals:
        t0 = time.time()
        g_tol = chebfun3(
            lambda x, y, z, _tol=tol: jnp.exp(jnp.sin(10 * x * y * z + jnp.exp(x * y * z))),
            tol=tol
        )
        elapsed = time.time() - t0
        I_tol = float(g_tol.sum3())
        err = abs(I_tol - I_g_full)
        results.append({
            "tol": tol,
            "rank": g_tol.rank,
            "I": I_tol,
            "err": err,
            "time": elapsed,
        })
        print(f"  tol={tol:.0e}: rank={g_tol.rank}, I={I_tol:.8f}, "
              f"err={err:.2e}, time={elapsed:.3f}s")

    # Looser tolerances should give fewer terms
    assert results[-1]["rank"][0] <= results[0]["rank"][0], \
        "Looser tolerance should give smaller or equal rank"

    # ------------------------------------------------------------------
    # Section 4: Coefficient decay visualization
    # ------------------------------------------------------------------
    print("\n--- Chebyshev coefficient decay ---")
    # For the simple function f, check Chebyshev coefficients of first fiber
    # (cols = list of Chebtech2)
    col0 = f.cols[0]  # First column Chebtech2
    coeffs0 = np.abs(np.array(col0.coeffs))
    print(f"  First column: {len(coeffs0)} Chebyshev coefficients")
    print(f"  Max coeff: {coeffs0.max():.6e}, Min coeff: {coeffs0.min():.6e}")
    # Coefficients should decay
    assert coeffs0[-1] < coeffs0[0] * 1e-5, "Coefficients should decay"

    # ------------------------------------------------------------------
    # Plot: tolerance vs rank and error
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3)

    # Tucker rank vs tolerance
    ax1 = axes[0]
    tols = [r["tol"] for r in results]
    ranks_x = [r["rank"][0] for r in results]
    ax1.loglog(tols, ranks_x, "o-b", lw=2, ms=10)
    ax1.invert_xaxis()
    ax1.set_title("Rank decreases with\nloose tolerance", fontsize=11)
    # Error vs tolerance
    ax2 = axes[1]
    errs = [r["err"] for r in results]
    ax2.loglog(tols, errs, "o-r", lw=2, ms=10)
    ax2.plot([1e-4, 1e-8], [1e-4, 1e-8], "--k", alpha=0.5, label="error=tol")
    ax2.invert_xaxis()
    ax2.set_title("Error scales with tolerance", fontsize=11)
    ax2.legend()
    # Chebyshev coefficients of first fiber
    ax3 = axes[2]
    ax3.semilogy(range(len(coeffs0)), coeffs0, "o-b", ms=4, lw=1.5)
    ax3.set_title("Coefficient decay of\nfirst column fiber", fontsize=11)
    fig.suptitle("Effect of tolerance on Chebfun3 construction", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        os.path.join(_IMG_DIR, "Tolerance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
