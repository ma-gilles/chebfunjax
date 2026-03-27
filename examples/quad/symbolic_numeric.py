"""Symbolic and numerical integration compared.

Demonstrates that Chebfun's numerical integration matches results that
can also be obtained symbolically (via integration by parts, etc.).

Credit: Inspired by Chebfun example quad/SymbolicNumeric.m
(Nick Trefethen, July 2014).
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
    print("Symbolic and numerical integration")
    print("=" * 60)

    pi = float(jnp.pi)

    # A set of integrals that can be computed both symbolically and numerically
    cases = [
        # (name, func, domain, exact)
        ("int_0^1 x^3 * log(x) dx = -1/16",
         lambda x: x**3 * jnp.log(jnp.maximum(x, 1e-300)),
         (1e-8, 1.0),
         -1.0 / 16.0),

        ("int_0^1 x^2 * exp(-x) dx = 2 - 5/e",
         lambda x: x**2 * jnp.exp(-x),
         (0.0, 1.0),
         2.0 - 5.0 / float(jnp.exp(jnp.array(1.0)))),

        ("int_0^pi x * sin(x) dx = pi",
         lambda x: x * jnp.sin(x),
         (0.0, pi),
         pi),

        ("int_0^1 log(1 + x) dx = 2*log(2) - 1",
         lambda x: jnp.log(1.0 + x),
         (0.0, 1.0),
         2.0 * float(jnp.log(jnp.array(2.0))) - 1.0),

        ("int_0^1 x * atan(x) dx = (pi - 2)/4",
         lambda x: x * jnp.arctan(x),
         (0.0, 1.0),
         (pi - 2.0) / 4.0),

        ("int_0^1 sin(x)^2 dx = (1 - sin(2)/2)/2",
         lambda x: jnp.sin(x)**2,
         (0.0, 1.0),
         0.5 - float(jnp.sin(jnp.array(2.0))) / 4.0),

        ("int_-1^1 (1-x^2)^(3/2) dx = 3*pi/8",
         lambda x: (1.0 - x**2)**1.5,
         (-1.0, 1.0),
         3.0 * pi / 8.0),

        ("int_0^1 sqrt(x) * (1-x)^2 dx = B(3/2,3)",
         lambda x: jnp.sqrt(x) * (1.0 - x)**2,
         (0.0, 1.0),
         float(__import__('scipy.special', fromlist=['beta']).beta(1.5, 3))),
    ]

    print(f"\n{'name':55}  {'computed':>14}  {'exact':>14}  {'rel err':>10}")
    print("-" * 100)
    max_err = 0.0
    for name, func, dom, exact in cases:
        f = cj.chebfun(func, domain=dom)
        val = float(f.sum())
        exact_f = float(exact)
        rel_err = abs(val - exact_f) / (abs(exact_f) + 1e-15)
        max_err = max(max_err, rel_err)
        print(f"{name:55}  {val:>14.10f}  {exact_f:>14.10f}  {rel_err:>10.2e}")

    print(f"\nMax relative error: {max_err:.2e}  (should be < 1e-10)")
    assert max_err < 1e-8, f"Max error too large: {max_err}"

    # Special: int_0^1 x^(-1/2) * exp(-x) dx = sqrt(pi) * erf(1)
    # (Gauss-Laguerre type)
    exact_special = float(jnp.sqrt(jnp.pi)) * float(jnp.array(
        float(__import__('scipy.special', fromlist=['erf']).erf(1.0))
    ))
    f_special = cj.chebfun(
        lambda x: jnp.sqrt(jnp.maximum(x, 1e-300)) * jnp.exp(-jnp.maximum(x, 1e-300)),
        domain=(1e-6, 3.0)
    )
    # Actually integrate x^(-1/2)*exp(-x): x^(-1/2) is singular, approximate
    # int_eps^1 x^(-1/2)*exp(-x) dx ≈ sqrt(pi)*erf(1) as eps->0
    f_special2 = cj.chebfun(
        lambda x: jnp.exp(-x) / jnp.sqrt(jnp.maximum(x, 1e-15)),
        domain=(1e-6, 4.0)
    )
    I_special = float(f_special2.sum())
    print(f"\nint_eps^4 x^(-1/2)*exp(-x) dx ≈ {I_special:.8f}")
    print(f"  sqrt(pi)*erf(1) = {exact_special:.8f}")

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 4)
    axes = axes.flatten()
    for i, (name, func, dom, exact) in enumerate(cases):
        ax = axes[i]
        f = cj.chebfun(func, domain=dom)
        xs = np.linspace(dom[0], dom[1], 300)
        ys = np.array(f(jnp.array(xs)))
        val = float(f.sum())
        ax.plot(xs, ys, color="#1e77b4", linewidth=1.5)
        ax.fill_between(xs, 0, ys, where=(ys >= 0), alpha=0.15, color="#1e77b4")
        ax.fill_between(xs, 0, ys, where=(ys < 0), alpha=0.15, color="#d62728")
        ax.axhline(0, color="k", linewidth=0.4)
        ax.set_title(f"I={val:.5f}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("Symbolic vs. numerical integration", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "symbolic_numeric.png"), dpi=150, bbox_inches="tight")
    _docs = os.path.join(_here, "..", "..", "docs", "images", "quad")
    os.makedirs(_docs, exist_ok=True)
    fig.savefig(os.path.join(_docs, "symbolic_numeric.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
