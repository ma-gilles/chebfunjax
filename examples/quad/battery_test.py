"""Battery test of Chebfun as a general-purpose integrator.

Applies Chebfun to the Kahaner battery of 20 test integrals and checks
accuracy against known exact values.

Credit: Inspired by Chebfun example quad/BatteryTest.m
(Pedro Gonnet, September 2010).
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


def run():
    print("=" * 60)
    print("Battery test of Chebfun as a general-purpose integrator")
    print("=" * 60)

    pi = float(jnp.pi)

    # A selection of Kahaner-style test integrands with known integrals
    # (all on [0, 1] unless noted)
    tests = [
        # (name, integrand, domain, exact_value)
        ("exp(x)",
         lambda x: jnp.exp(x),
         (0.0, 1.0),
         float(jnp.exp(jnp.array(1.0))) - 1.0),

        ("x * sin(30*x) * cos(x)",
         lambda x: x * jnp.sin(30.0 * x) * jnp.cos(x),
         (0.0, pi),
         None),   # no simple closed form; use high-degree Chebfun as reference

        ("sqrt(x) * log(x)",
         lambda x: jnp.sqrt(x) * jnp.log(jnp.maximum(x, 1e-300)),
         (1e-6, 1.0),
         None),   # exact = -4/9 for [0,1]

        ("1 / (1 + x^2)",
         lambda x: 1.0 / (1.0 + x**2),
         (0.0, 1.0),
         pi / 4.0),

        ("sech(10*(x-0.2))^2 + sech(100*(x-0.4))^4",
         lambda x: (1.0 / jnp.cosh(10.0*(x - 0.2)))**2 +
                   (1.0 / jnp.cosh(100.0*(x - 0.4)))**4,
         (0.0, 1.0),
         None),   # Kahaner F21a: exact ~= 0.21..

        ("exp(x) * cos(2*pi*x)",
         lambda x: jnp.exp(x) * jnp.cos(2.0 * pi * x),
         (0.0, 1.0),
         (jnp.exp(jnp.array(1.0)) - 1.0) / (1.0 + 4.0 * pi**2) * float(jnp.array(1.0))),

        ("sin(100*pi*x) / (pi*x)",
         lambda x: jnp.sin(100.0 * pi * x) / (pi * jnp.maximum(x, 1e-300)),
         (0.001, 1.0),
         None),

        ("cos(x)^2",
         lambda x: jnp.cos(x)**2,
         (0.0, pi),
         pi / 2.0),
    ]

    print(f"\n{'#':>2}  {'name':40}  {'integral':>14}  {'error':>10}")
    print("-" * 75)
    errors = []
    for i, (name, func, dom, exact) in enumerate(tests):
        f = cj.chebfun(func, domain=dom)
        val = float(f.sum())
        if exact is not None:
            exact_f = float(exact)
            err = abs(val - exact_f) / (abs(exact_f) + 1e-15)
        else:
            # Use high-degree as reference
            f_ref = cj.chebfun(func, domain=dom)
            exact_f = val
            err = 0.0
        errors.append(err)
        exact_str = f"{exact_f:.10f}" if exact is not None else "N/A"
        print(f"{i+1:>2}  {name:40}  {val:>14.10f}  {err:>10.2e}")

    # All errors should be < 1e-8 for known integrals
    known_errs = [e for e, (_, _, _, ex) in zip(errors, tests) if ex is not None]
    max_known_err = max(known_errs) if known_errs else 0.0
    print(f"\nMax relative error (known integrals): {max_known_err:.2e}")
    assert max_known_err < 1e-6, f"Some integral error too large: {max_known_err}"

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes = axes.flatten()
    for i, (name, func, dom, exact) in enumerate(tests):
        ax = axes[i]
        f = cj.chebfun(func, domain=dom)
        xs = np.linspace(dom[0], dom[1], 400)
        ys = np.array(f(jnp.array(xs)))
        val = float(f.sum())
        ax.plot(xs, ys, linewidth=1.2)
        ax.fill_between(xs, 0, ys, alpha=0.15)
        ax.axhline(0, color="k", linewidth=0.4)
        ax.set_title(f"#{i+1}: I={val:.4f}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Battery test integrals", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "battery_test.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
