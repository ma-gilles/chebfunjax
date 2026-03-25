"""Chebyshev polynomials.

Demonstrates Chebyshev polynomial evaluation, extrema, and classical plots,
following cheb/ChebPolysHigham.m by Nick Trefethen (December 2011).

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
from chebfunjax.utils.polynomials import chebpoly


def eval_chebpoly(n, xs):
    """Evaluate T_n at array xs by building a chebfun."""
    coeffs = chebpoly(n)
    f = cj.chebfun.from_coeffs(coeffs)
    return np.array([float(f(jnp.array(float(x)))) for x in np.asarray(xs)])


def run():
    print("=" * 60)
    print("Chebyshev polynomials")
    print("=" * 60)

    # Evaluate T_n at Chebyshev nodes (should equal standard cos formula)
    xs = np.linspace(-0.99, 0.99, 100)

    print("\nT_n(cos(θ)) = cos(nθ):")
    for n in [1, 3, 5]:
        Tn_vals = eval_chebpoly(n, xs)
        # T_n(x) = cos(n * arccos(x))
        Tn_exact = np.cos(n * np.arccos(np.clip(xs, -1, 1)))
        err = np.max(np.abs(Tn_vals - Tn_exact))
        print(f"  T_{n}: max error = {err:.2e}")
        assert err < 1e-10, f"T_{n} error too large"

    # T_n(1) = 1 for all n
    for n in range(6):
        fn = cj.chebfun.from_coeffs(chebpoly(n))
        Tn_at_1 = float(fn(jnp.array(1.0)))
        assert abs(Tn_at_1 - 1.0) < 1e-10, f"T_{n}(1) != 1, got {Tn_at_1}"
    print("T_n(1) = 1 for n = 0,...,5  [OK]")

    # T_n(-1) = (-1)^n
    for n in range(6):
        fn = cj.chebfun.from_coeffs(chebpoly(n))
        Tn_at_m1 = float(fn(jnp.array(-1.0)))
        expected = (-1)**n
        assert abs(Tn_at_m1 - expected) < 1e-10, f"T_{n}(-1) != {expected}, got {Tn_at_m1}"
    print("T_n(-1) = (-1)^n for n = 0,...,5  [OK]")

    # Orthogonality: integral T_m * T_n with Clenshaw-Curtis weights
    n_quad = 256
    from chebfunjax.utils.quadrature import chebpts, chebweights
    xs_q = chebpts(n_quad)
    ws_q = chebweights(n_quad)
    xs_q_np = np.array(xs_q)
    ws_q_np = np.array(ws_q)

    # Chebyshev orthogonality with standard L2 weight (not 1/sqrt(1-x^2)):
    # <T_m, T_n>_L2 = 0 for m != n (from orthogonality over the nodes)
    # <T_0, T_0> = 2, <T_n, T_n> = 1 for n>=1 (with Chebyshev measure)
    # With Clenshaw-Curtis weights on [-1,1]:
    # <T_m, T_n>_CC = integral T_m T_n dx
    # T_0 = 1: int T_0^2 = 2; T_1 = x: int x^2 = 2/3; T_2 = 2x^2-1: etc.
    # Verify just orthogonality (m != n)
    for m, n_idx in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        Tm = eval_chebpoly(m, xs_q_np)
        Tn_v = eval_chebpoly(n_idx, xs_q_np)
        inner = float(np.dot(ws_q_np, Tm * Tn_v))
        print(f"  <T_{m}, T_{n_idx}> = {inner:.6f}  (expected: 0 by orthogonality)")
        assert abs(inner) < 0.01, f"Orthogonality failed for T_{m}, T_{n_idx}: {inner}"

    # Plot
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in range(7):
        Tn = eval_chebpoly(n, xs)
        ax.plot(xs, Tn, label=f"T_{n}")
    ax.set_xlim(-1, 1); ax.set_ylim(-1.2, 1.2)
    ax.set_title("Chebyshev polynomials T_0, ..., T_6", fontsize=13)
    ax.set_xlabel("x"); ax.set_ylabel("T_n(x)")
    ax.legend(ncol=2, fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "chebyshev_polynomials.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
