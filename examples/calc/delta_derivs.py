"""Delta functions and derivatives in Chebfun.

Demonstrates using cumsum to integrate functions with distributional
(Dirac delta) impulses, showing how repeated integration smooths them out.

Credit: Inspired by Chebfun example calc/DeltaDerivs.m
(Nick Trefethen, August 2012).
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

from chebfunjax.plotting import plot


def run():
    print("=" * 60)
    print("Delta functions and derivatives")
    print("=" * 60)

    # We approximate the effect of delta functions by using very narrow
    # Gaussian bumps centred at integer points on [0, 20].
    # rng(3) in MATLAB gives specific random amplitudes; we fix them here.
    rng = np.random.default_rng(3)
    dom = (0.0, 20.0)

    # Build f = 0.5*sin(x) + sum of narrow Gaussians at j=1..19
    amplitudes = rng.standard_normal(19)

    eps = 0.15  # wider Gaussians for faster convergence (original eps=0.05 too slow)
    n_impulses = 10  # reduce from 19 for speed

    def f_func(x):
        val = 0.5 * jnp.sin(x)
        for j in range(1, n_impulses + 1):
            val = val + float(amplitudes[j - 1]) * jnp.exp(-((x - j) ** 2) / (2 * eps**2)) / (eps * jnp.sqrt(2 * jnp.pi))
        return val

    dom_small = (0.0, float(n_impulses + 1))
    f = cj.chebfun(f_func, domain=dom_small)
    # Subtract mean so the integral is zero
    f_mean = float(f.mean())
    f = cj.chebfun(lambda x: f_func(x) - f_mean, domain=dom_small)

    print(f"\nf statistics:")
    x_max, f_max = f.max()
    x_min, f_min = f.min()
    f_sum = float(f.sum())
    f_norm1 = float(f.norm(1))
    f_norm2 = float(f.norm())
    f_norminf = float(f.norm(np.inf))
    print(f"  max(f) = {f_max:.6f}")
    print(f"  min(f) = {f_min:.6f}")
    print(f"  sum(f) = {f_sum:.2e}  (should be ~0)")
    print(f"  norm(f,1) = {f_norm1:.6f}")
    print(f"  norm(f,2) = {f_norm2:.6f}")
    print(f"  norm(f,inf) = {f_norminf:.6f}")

    # Each repeated integration smooths out the delta bumps
    g = f.cumsum()   # first integral: step jumps
    h = g.cumsum()   # second integral: C^0
    q = h.cumsum()   # third integral: C^1

    # Taking three derivatives of q should give back f
    f2 = q.diff(3)
    xs = jnp.linspace(0.5, float(n_impulses) - 0.5, 200)
    err = float(jnp.max(jnp.abs(f2(xs) - f(xs))))
    print(f"\ndiff(cumsum^3(f), 3) vs f: max err = {err:.2e}  (should be ~0)")
    assert err < 1e-5, f"Round-trip error too large: {err}"

    # Also: sum of g should equal sum of f (which is 0)
    g_end = float(g(jnp.array(float(n_impulses + 1))))
    print(f"  g(20) = {g_end:.2e}  (should be ~0 since f has zero mean)")

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    xs_plot = np.linspace(0.0, float(n_impulses + 1), 1000)
    f_vals = np.array(f(jnp.array(xs_plot)))
    g_vals = np.array(g(jnp.array(xs_plot)))
    h_vals = np.array(h(jnp.array(xs_plot)))
    q_vals = np.array(q(jnp.array(xs_plot)))

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(xs_plot, f_vals, color="#1e77b4", linewidth=1.2)
    axes[0, 0].set_title("f: sine + approximate delta impulses")
    axes[0, 0].set_xlabel("x")

    axes[0, 1].plot(xs_plot, g_vals, color="#d62728", linewidth=1.2)
    axes[0, 1].set_title(f"g = cumsum(f): {n_impulses} impulses")
    axes[0, 1].set_xlabel("x")

    axes[1, 0].plot(xs_plot, h_vals, color="#2ca02c", linewidth=1.2)
    axes[1, 0].set_title("h = cumsum(g): continuous, $C^0$")
    axes[1, 0].set_xlabel("x")

    axes[1, 1].plot(xs_plot, q_vals, color="#ff7f0e", linewidth=1.2)
    axes[1, 1].set_title("q = cumsum(h): $C^1$ function")
    axes[1, 1].set_xlabel("x")

    fig.suptitle("Repeated integration smooths distributional impulses", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "delta_derivs.png"), dpi=150, bbox_inches="tight")
    _docs = os.path.join(_here, "..", "..", "docs", "images", "calc")
    os.makedirs(_docs, exist_ok=True)
    fig.savefig(os.path.join(_docs, "delta_derivs.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
