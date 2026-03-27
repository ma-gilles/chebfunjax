"""Chebfun timings and Tucker rank complexity for 1D, 2D, and 3D functions.

Illustrates how the number of terms in a Chebfun3 approximation (its Tucker
rank) grows with the difficulty of the function, comparing smooth/low-rank
functions (Runge) with hard/full-rank functions (tanh).

Original MATLAB Chebfun: approx3/Complexity.m by Nick Trefethen, April 2015.
See https://www.chebfun.org/examples/approx3/Complexity.html
Copyright 2015 by The University of Oxford and The Chebfun Developers.
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
    print("Chebfun3 complexity: Tucker rank vs function difficulty")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Section 1: Hard 3D function — tanh(k*(x+y+z)/sqrt(3))
    # ------------------------------------------------------------------
    print("\n--- Hard 3D: tanh(k*(x+y+z)/sqrt(3)) ---")
    kk_hard = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    ranks_hard = []
    times_hard = []

    for k in kk_hard:
        t0 = time.time()
        f = chebfun3(
            lambda x, y, z, _k=k: jnp.tanh(_k * (x + y + z) / np.sqrt(3))
        )
        elapsed = time.time() - t0
        rx, ry, rz = f.rank
        ranks_hard.append(rx)
        times_hard.append(elapsed)
        print(f"  k={k:4.1f}: rank=({rx},{ry},{rz}), time={elapsed:.3f}s")

    # ------------------------------------------------------------------
    # Section 2: Easy 3D function — 1/(1+k*(x^2+y^2+z^2)) (Runge)
    # ------------------------------------------------------------------
    print("\n--- Easy 3D (Runge): 1/(1+k*(x^2+y^2+z^2)) ---")
    kk_easy = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    ranks_easy = []
    times_easy = []

    for k in kk_easy:
        t0 = time.time()
        f = chebfun3(
            lambda x, y, z, _k=k: 1.0 / (1.0 + _k * (x**2 + y**2 + z**2))
        )
        elapsed = time.time() - t0
        rx, ry, rz = f.rank
        ranks_easy.append(rx)
        times_easy.append(elapsed)
        print(f"  k={k:5.1f}: rank=({rx},{ry},{rz}), time={elapsed:.3f}s")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    print("\n--- Validation: accuracy checks ---")

    # Exact integral for Runge k=1: int_{-1}^1^3 1/(1+x^2+y^2+z^2) dx dy dz
    # Exact value (scipy tplquad): 4.286854...
    f_k1 = chebfun3(lambda x, y, z: 1.0 / (1.0 + x**2 + y**2 + z**2))
    I_k1 = float(f_k1.sum3())
    exact_k1 = 4.286854062301845
    print(f"  int 1/(1+r^2) over [-1,1]^3 = {I_k1:.8f}  (exact: {exact_k1:.8f})")
    assert abs(I_k1 - exact_k1) / exact_k1 < 1e-5

    # tanh is odd, so integral over symmetric domain = 0
    f_tanh = chebfun3(lambda x, y, z: jnp.tanh(x + y + z))
    I_tanh = float(f_tanh.sum3())
    print(f"  int tanh(x+y+z) over [-1,1]^3 = {I_tanh:.2e}  (exact: 0)")
    assert abs(I_tanh) < 1e-8

    # ------------------------------------------------------------------
    # Plot: rank vs k for both functions
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2)

    ax1 = axes[0]
    ax1.semilogy(kk_hard, ranks_hard, "ob-", lw=2, ms=8)
    ax1.set_title("Hard: tanh(k(x+y+z)/√3)\nTucker rank grows with k", fontsize=11)
    ax2 = axes[1]
    ax2.semilogy(kk_easy, ranks_easy, "or-", lw=2, ms=8)
    ax2.set_title("Easy: 1/(1+k·r²) (Runge)\nLow Tucker rank for all k", fontsize=11)
    fig.suptitle("Chebfun3 Tucker rank complexity", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        os.path.join(_IMG_DIR, "Complexity.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
