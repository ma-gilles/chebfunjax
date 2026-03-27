"""Difficult integrals via Chebfun.

Demonstrates that Chebfun can handle highly oscillatory, nearly
singular, and other difficult integrands to high precision.

Credit: Inspired by Chebfun examples quad/TrickyIntegrands.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.plotting import plot


def run():
    print("=" * 60)
    print("Difficult integrals via Chebfun")
    print("=" * 60)

    # --- 1. High-frequency oscillation: sin(100*x) on [-1, 1] -------
    dom = (-1.0, 1.0)
    f1 = cj.chebfun(lambda x: jnp.sin(100.0 * x), domain=dom)
    int1 = float(f1.sum())
    print(f"\n1. sin(100x) on [-1,1]:")
    print(f"   Computed: {int1:.2e}  (exact: 0)")
    assert abs(int1) < 1e-12

    # --- 2. Highly oscillatory with exponential: exp(sin(50x)) ------
    dom2 = (0.0, float(2.0 * jnp.pi))
    f2 = cj.chebfun(lambda x: jnp.exp(jnp.sin(50.0 * x)), domain=dom2)
    int2 = float(f2.sum())
    exact2, _ = quad(lambda x: float(np.exp(np.sin(50.0 * x))),
                     0.0, float(2.0 * np.pi))
    err2 = abs(int2 - exact2)
    print(f"\n2. exp(sin(50x)) on [0, 2*pi]:")
    print(f"   Computed: {int2:.10f}")
    print(f"   Scipy:    {exact2:.10f}")
    print(f"   Error:    {err2:.2e}")
    assert err2 < 1e-8

    # --- 3. Product of high-frequency functions ----------------------
    dom3 = (0.0, 1.0)
    f3 = cj.chebfun(lambda x: jnp.sin(100.0 * jnp.pi * x) * jnp.cos(100.0 * jnp.pi * x), domain=dom3)
    int3 = float(f3.sum())
    print(f"\n3. sin(100*pi*x)*cos(100*pi*x) on [0,1]:")
    print(f"   Computed: {int3:.2e}  (exact: 0)")
    assert abs(int3) < 1e-11

    # --- 4. Square root singularity (integrable): 1/sqrt(1-x^2) -----
    dom4 = (-0.999, 0.999)  # avoid exact endpoints
    f4 = cj.chebfun(lambda x: 1.0 / jnp.sqrt(1.0 - x**2), domain=dom4)
    int4 = float(f4.sum())
    exact4 = 2.0 * float(jnp.arcsin(jnp.array(0.999)))
    err4 = abs(int4 - exact4)
    print(f"\n4. 1/sqrt(1-x^2) on [-0.999, 0.999]:")
    print(f"   Computed: {int4:.10f}")
    print(f"   Exact (2*arcsin(0.999)): {exact4:.10f}")
    print(f"   Error: {err4:.2e}")
    assert err4 < 1e-8

    # --- 5. Spike integral: exp(-1000*x^2) -------------------------
    dom5 = (-1.0, 1.0)
    f5 = cj.chebfun(lambda x: jnp.exp(-1000.0 * x**2), domain=dom5)
    int5 = float(f5.sum())
    exact5 = float(jnp.sqrt(jnp.pi / 1000.0))
    err5 = abs(int5 - exact5)
    print(f"\n5. exp(-1000*x^2) on [-1,1]:")
    print(f"   Computed: {int5:.12f}")
    print(f"   sqrt(pi/1000): {exact5:.12f}")
    print(f"   Error: {err5:.2e}")
    assert err5 < 1e-10

    # --- 6. Logistic function ------------------------------------
    dom6 = (-10.0, 10.0)
    f6 = cj.chebfun(lambda x: 1.0 / (1.0 + jnp.exp(-x)), domain=dom6)
    int6 = float(f6.sum())
    exact6 = float(jnp.log(1.0 + jnp.exp(jnp.array(10.0))) -
                    jnp.log(1.0 + jnp.exp(jnp.array(-10.0))))
    err6 = abs(int6 - exact6)
    print(f"\n6. 1/(1+exp(-x)) on [-10, 10]:")
    print(f"   Computed: {int6:.12f}")
    print(f"   Exact: {exact6:.12f}")
    print(f"   Error: {err6:.2e}")
    assert err6 < 1e-10

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f1, title="sin(100x) — highly oscillatory")
    fig.savefig(os.path.join(_here, "tricky_integrals.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
