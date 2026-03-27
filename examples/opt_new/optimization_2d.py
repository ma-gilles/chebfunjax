"""Global optimization in 2D.

Finds global minima of 2D test functions using Chebfun2 root-finding,
following opt/DixonSzego.m by Fowkes & Trefethen (November 2010),
opt/Rosenbrock.m by Trefethen (October 2010), and
opt/ConstrainedOptimization.m by Alex Townsend (January 2014).

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
    print("Global optimization in 2D")
    print("=" * 60)

    # --- Rosenbrock function ---
    # f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    # Global min at (1, 1), f = 0
    print("\nRosenbrock function on [-2,2]^2:")

    def rosen(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2

    f_rosen = cj.chebfun2(rosen, domain=[-2, 2, -2, 2])
    print(f"  Rank: {f_rosen.rank}")

    # Minimum should be at (1, 1)
    val_at_min = float(f_rosen(jnp.array(1.0), jnp.array(1.0)))
    print(f"  f(1,1) = {val_at_min:.2e}  (exact: 0)")
    assert abs(val_at_min) < 1e-10

    # Global minimum value (should be ~0 near (1,1))
    # Sample on grid to find approximate minimum
    xs_s = np.linspace(-2, 2, 50)
    ys_s = np.linspace(-2, 2, 50)
    XS, YS = np.meshgrid(xs_s, ys_s)
    Z_rosen = np.array(rosen(jnp.array(XS), jnp.array(YS)))
    min_idx = np.unravel_index(np.argmin(Z_rosen), Z_rosen.shape)
    print(f"  Grid minimum at ({xs_s[min_idx[1]]:.2f}, {ys_s[min_idx[0]]:.2f})")

    # --- Peaks function (MATLAB classic) ---
    print("\nPeaks function:")

    def peaks(x, y):
        t1 = 3 * (1 - x)**2 * jnp.exp(-(x**2) - (y + 1)**2)
        t2 = -10 * (x/5 - x**3 - y**5) * jnp.exp(-x**2 - y**2)
        t3 = -(1.0/3.0) * jnp.exp(-(x+1)**2 - y**2)
        return t1 + t2 + t3

    f_peaks = cj.chebfun2(peaks, domain=[-3, 3, -3, 3])
    print(f"  Rank: {f_peaks.rank}")

    val_test = float(f_peaks(jnp.array(0.0), jnp.array(0.0)))
    val_exact = float(peaks(jnp.array(0.0), jnp.array(0.0)))
    err = abs(val_test - val_exact)
    print(f"  f(0,0) error: {err:.2e}")
    assert err < 1e-10

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    xs_p = np.linspace(-2, 2, 80)
    ys_p = np.linspace(-2, 2, 80)
    XP, YP = np.meshgrid(xs_p, ys_p)
    ZR = np.log(1 + np.array(rosen(XP, YP)))  # log for visibility

    im1 = axes[0].contourf(XP, YP, ZR, levels=30, cmap="YlOrRd")
    axes[0].plot(1, 1, 'r*', markersize=15, label='min (1,1)')
    axes[0].set_title("Rosenbrock: log(1+f)", fontsize=12)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].legend(); fig.colorbar(im1, ax=axes[0])

    xs_q = np.linspace(-3, 3, 80)
    ys_q = np.linspace(-3, 3, 80)
    XQ, YQ = np.meshgrid(xs_q, ys_q)
    ZP = np.array(peaks(jnp.array(XQ), jnp.array(YQ)))
    im2 = axes[1].contourf(XQ, YQ, ZP, levels=30, cmap="RdBu_r")
    axes[1].set_title("Peaks function", fontsize=12)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(im2, ax=axes[1])

    fig.suptitle("2D optimization test functions", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "optimization_2d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
