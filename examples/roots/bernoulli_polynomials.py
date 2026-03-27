"""Bernoulli polynomials: roots, convergence, and Bernoulli numbers.

Demonstrates constructing Bernoulli polynomials by repeated integration,
verifying that each polynomial has at most 3 real roots on [0,1], and
showing the geometric convergence of rescaled odd Bernoulli polynomials to sin.

Credit: Inspired by Chebfun example roots/BernoulliPolynomials.m
(Stefan Guettel, February 2012).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def run():
    print("=" * 60)
    print("Bernoulli polynomials")
    print("=" * 60)

    dom = (0.0, 1.0)

    # Build Bernoulli polynomials recursively:
    # B_0(x) = 1
    # B_j(x) = j * integral of B_{j-1}(x) dx, with constant chosen so
    #          int_0^1 B_j(x) dx = 0 (i.e. B_j has zero mean).
    N = 20
    B = []
    b0 = cj.chebfun(lambda x: jnp.ones_like(x), domain=dom)
    B.append(b0)
    for j in range(1, N + 1):
        bj = cj.chebfun(lambda x, _j=j, _prev=B[j-1]: _j * _prev.cumsum()(jnp.array(x)) - _j * float(_prev.cumsum().mean()), domain=dom)
        B.append(bj)

    # Print Bernoulli numbers B_j(0) for j = 0..13
    print("\nBernoulli numbers B_j(0) for j = 0..13:")
    bn = []
    for j in range(14):
        val = float(B[j](jnp.array(0.0)))
        bn.append(val)
        print(f"  B_{j}(0) = {val:.8f}")

    # Known Bernoulli numbers
    exact_bn = [1, -0.5, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66, 0, -691/2730, 0]
    for j in range(14):
        err = abs(bn[j] - exact_bn[j])
        assert err < 1e-6, f"B_{j}(0) error: {err}"
    print("All Bernoulli numbers match to 1e-6.")

    # Count roots of each Bernoulli polynomial on [0,1]
    print("\nNumber of real roots of B_j on [0,1] for j = 1..20:")
    max_roots = 0
    for j in range(1, N + 1):
        r = B[j].roots()
        nr = len(r)
        max_roots = max(max_roots, nr)
        if j <= 10:
            print(f"  B_{j}: {nr} roots")
    print(f"  Maximum number of roots (j=1..{N}): {max_roots}")
    assert max_roots <= 3, f"Expected at most 3 roots, got {max_roots}"

    # Convergence of rescaled odd Bernoulli polynomials to sin(2*pi*x)
    limit = cj.chebfun(lambda x: jnp.sin(2.0 * jnp.pi * x), domain=dom)
    errs = []
    js = range(1, min(N // 2 + 1, 10))
    for j in js:
        sign = (-1)**j
        scale = (2 * math.pi)**(2*j - 1) / 2.0 / math.factorial(2*j - 1)
        bj_scaled = cj.chebfun(lambda x, _j=j, _s=sign*scale, _bj=B[2*j]: _s * _bj(jnp.array(x)), domain=dom)
        err = float((bj_scaled - limit).norm(np.inf))
        errs.append(err)
    print("\nConvergence of rescaled odd B_{2j} polynomials to sin(2*pi*x):")
    for i, j in enumerate(js):
        print(f"  j={j}: err = {errs[i]:.2e}")

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    xs = np.linspace(0.0, 1.0, 400)

    fig, axes = plt.subplots(1, 2)

    # Left: first 6 Bernoulli polynomials
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    for j in range(1, 7):
        ys = np.array(B[j](jnp.array(xs)))
        axes[0].plot(xs, ys, color=colors[j-1], linewidth=1.5, label=f"$B_{j}$")
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-0.3, 0.3)
    axes[0].set_title("First 6 Bernoulli polynomials on [0,1]")
    axes[0].legend(fontsize=8, ncol=2)

    # Right: convergence to sin
    if errs:
        axes[1].semilogy(list(js), errs, 'b.-', markersize=8, linewidth=1.5, label="error")
        axes[1].semilogy(list(js), [0.5**j for j in js], 'r--', linewidth=1.2, label="$0.5^j$ (rate)")
        axes[1].set_title("Geometric convergence to sin")
        axes[1].legend(fontsize=9)

    fig.suptitle("Bernoulli polynomials: roots and convergence", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "bernoulli_polynomials.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
