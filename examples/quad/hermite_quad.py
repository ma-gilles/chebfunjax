"""Hermite (Gauss-Hermite) quadrature.

Gauss-Hermite quadrature integrates functions of the form exp(-x^2)*f(x)
on (-inf, inf).  We demonstrate convergence by comparing with Chebfun
quadrature on a large finite interval.

Credit: Inspired by Chebfun example quad/HermiteQuad.m
(Nick Trefethen and Andre Weideman, February 2017).
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
    print("Hermite quadrature")
    print("=" * 60)

    pi = float(jnp.pi)

    # Gauss-Hermite nodes and weights: exact for degree <= 2n-1 polynomials
    # int_{-inf}^{inf} exp(-x^2) p(x) dx ~ sum w_j p(x_j)
    # Compute via eigenvalue method (symmetric tridiagonal)
    def hermite_nodes_weights(n):
        """Gauss-Hermite nodes and weights via eigenvalue method."""
        k = np.arange(1, n)
        beta = np.sqrt(k / 2.0)
        T = np.diag(beta, 1) + np.diag(beta, -1)
        vals, vecs = np.linalg.eigh(T)
        idx = np.argsort(vals)
        x = vals[idx]
        w = np.sqrt(pi) * vecs[0, idx]**2
        return x, w

    # Test 1: int exp(-x^2) dx = sqrt(pi)
    # This is just sum of weights
    exact_sqrt_pi = float(jnp.sqrt(jnp.pi))
    print(f"\nTest 1: int_(-inf,inf) exp(-x^2) dx = sqrt(pi) = {exact_sqrt_pi:.10f}")
    for n in [5, 10, 20]:
        x, w = hermite_nodes_weights(n)
        I = float(np.sum(w))
        err = abs(I - exact_sqrt_pi)
        print(f"  n={n:3d}: I = {I:.14f}, err = {err:.2e}")
        if n >= 5:
            assert err < 1e-12, f"n={n}: error {err}"

    # Test 2: int exp(-x^2) * x^2 dx = sqrt(pi)/2
    exact_x2 = exact_sqrt_pi / 2.0
    print(f"\nTest 2: int_(-inf,inf) exp(-x^2) * x^2 dx = sqrt(pi)/2 = {exact_x2:.10f}")
    for n in [5, 10, 20]:
        x, w = hermite_nodes_weights(n)
        I = float(np.sum(w * x**2))
        err = abs(I - exact_x2)
        print(f"  n={n:3d}: I = {I:.14f}, err = {err:.2e}")
        assert err < 1e-12 * abs(exact_x2), f"n={n}: error {err}"

    # Test 3: int exp(-x^2) * cos(x) dx = sqrt(pi) * exp(-1/4)
    exact_cos = exact_sqrt_pi * float(jnp.exp(jnp.array(-0.25)))
    print(f"\nTest 3: int_(-inf,inf) exp(-x^2) * cos(x) dx = sqrt(pi)*exp(-1/4) = {exact_cos:.10f}")
    ns = [5, 10, 20, 30, 50]
    errs_cos = []
    for n in ns:
        x, w = hermite_nodes_weights(n)
        I = float(np.sum(w * np.cos(x)))
        err = abs(I - exact_cos)
        errs_cos.append(err)
        print(f"  n={n:3d}: I = {I:.14f}, err = {err:.2e}")
    assert errs_cos[-1] < 1e-12 * abs(exact_cos)

    # Chebfun comparison on large interval [-L, L]
    L = 6.0
    f_cheb = cj.chebfun(lambda x: jnp.exp(-x**2) * jnp.cos(x), domain=(-L, L))
    I_cheb = float(f_cheb.sum())
    err_cheb = abs(I_cheb - exact_cos)
    print(f"\nChebfun on [-{L},{L}]: I = {I_cheb:.14f}, err = {err_cheb:.2e}")
    assert err_cheb < 1e-12 * abs(exact_cos)

    # Test 4: int exp(-x^2) * exp(x) dx = sqrt(pi) * exp(1/4)
    exact_expx = exact_sqrt_pi * float(jnp.exp(jnp.array(0.25)))
    print(f"\nTest 4: int exp(-x^2) * exp(x) dx = sqrt(pi)*exp(1/4) = {exact_expx:.10f}")
    for n in [5, 10, 20]:
        x, w = hermite_nodes_weights(n)
        I = float(np.sum(w * np.exp(x)))
        err = abs(I - exact_expx)
        print(f"  n={n:3d}: I = {I:.14f}, err = {err:.2e}")
        # n=5 only gives a few digits; need n>=10 for 1e-12 accuracy
        if n >= 10:
            assert err < 1e-12 * abs(exact_expx), f"n={n}: error {err}"

    # --- Plots ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Left: convergence for Test 3
    axes[0].semilogy(ns, [max(e, 1e-16) for e in errs_cos], color='#0072BD', linestyle='.-', markersize=8, linewidth=1.5)
    axes[0].set_title("Hermite quadrature convergence\n$\\int e^{-x^2} \\cos x\\, dx$")

    # Right: the integrand exp(-x^2) * cos(x)
    xs_plot = np.linspace(-5, 5, 500)
    ys_plot = np.exp(-xs_plot**2) * np.cos(xs_plot)
    n_plot = 15
    xq, wq = hermite_nodes_weights(n_plot)
    axes[1].plot(xs_plot, ys_plot, color="#1e77b4", linewidth=1.8, label="$e^{-x^2}\\cos x$")
    axes[1].stem(xq, wq * np.cos(xq), linefmt="r-", markerfmt="ro",
                 basefmt="k-", label=f"$w_j \\cos(x_j)$, n={n_plot}")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_title("Hermite quadrature nodes and weights")
    axes[1].legend(fontsize=9)

    fig.suptitle("Gauss-Hermite quadrature", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "hermite_quad.png"), dpi=150, bbox_inches="tight")
    _docs = os.path.join(_here, "..", "..", "docs", "images", "quad")
    os.makedirs(_docs, exist_ok=True)
    fig.savefig(os.path.join(_docs, "hermite_quad.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
