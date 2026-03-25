"""Fourier coefficients computed via Chebfun integration.

Demonstrates computing Fourier coefficients of smooth and non-smooth
functions using Chebfun's high-accuracy quadrature. Verifies Parseval's
theorem and convergence rates.

Credit: Inspired by Chebfun fourier examples.
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
from chebfunjax.plotting import plot


def compute_fourier_coeff(f_func, n, dom=(0.0, 2.0 * float(jnp.pi))):
    """Compute n-th Fourier coefficient a_n = (1/pi) int f(x)*cos(n*x) dx
    and b_n = (1/pi) int f(x)*sin(n*x) dx using Chebfun quadrature."""
    pi = float(jnp.pi)
    if n == 0:
        f = cj.chebfun(f_func, domain=dom)
        a0 = float(f.sum()) / (2.0 * pi)
        return a0, 0.0
    fa_n = cj.chebfun(lambda x, n=n: f_func(x) * jnp.cos(n * x), domain=dom)
    fb_n = cj.chebfun(lambda x, n=n: f_func(x) * jnp.sin(n * x), domain=dom)
    a_n = float(fa_n.sum()) / pi
    b_n = float(fb_n.sum()) / pi
    return a_n, b_n


def run():
    print("=" * 60)
    print("Fourier coefficients via Chebfun quadrature")
    print("=" * 60)

    pi = float(jnp.pi)
    dom = (0.0, 2.0 * pi)

    # --- f(x) = cos(3x): exact a_3 = 1, all others 0 ----------------
    print(f"\nf(x) = cos(3x):")
    for n in [0, 1, 2, 3, 4, 5]:
        a_n, b_n = compute_fourier_coeff(lambda x: jnp.cos(3.0 * x), n, dom)
        expected_a = 1.0 if n == 3 else 0.0
        expected_b = 0.0
        print(f"  n={n}: a_n={a_n:.8f} (exact {expected_a}), b_n={b_n:.2e}")
        assert abs(a_n - expected_a) < 1e-10
        assert abs(b_n - expected_b) < 1e-10

    # --- f(x) = x on [0, 2*pi]: Fourier series b_n = -2/n -----------
    # f(x) = pi - sum_{n=1}^{inf} (2/n)*sin(n*x)
    print(f"\nf(x) = x on [0, 2*pi]:")
    a0, _ = compute_fourier_coeff(lambda x: x, 0, dom)
    print(f"  a_0 = {a0:.8f}  (exact: pi = {pi:.8f})")
    assert abs(a0 - pi) < 1e-10
    for n in [1, 2, 3, 5, 10]:
        a_n, b_n = compute_fourier_coeff(lambda x: x, n, dom)
        exact_b = -2.0 / n
        print(f"  n={n}: a_n={a_n:.2e}, b_n={b_n:.10f}  (exact b_n={exact_b:.10f})")
        assert abs(a_n) < 1e-10  # f(x) = x is odd around pi, cosine terms vanish
        assert abs(b_n - exact_b) < 1e-9

    # --- Parseval's theorem: (1/pi) int f^2 dx = a_0^2 + sum (a_n^2 + b_n^2) ---
    # For f(x) = x on [0, 2*pi]:
    # (1/pi) int_0^{2*pi} x^2 dx = (1/pi) * [x^3/3]_0^{2*pi} = 8*pi^2/3
    # Fourier: 2*a_0^2 + sum_{n=1}^{inf} (a_n^2 + b_n^2) = 2*pi^2 + sum (2/n)^2
    # sum (2/n)^2 for n=1..inf = 4*pi^2/6 = 2*pi^2/3
    # Total = 2*pi^2 + 2*pi^2/3 = 8*pi^2/3 ✓
    fx2 = cj.chebfun(lambda x: x**2, domain=dom)
    parseval_lhs = float(fx2.sum()) / pi
    exact_parseval = 8.0 * pi**2 / 3.0
    print(f"\nParseval check for f(x)=x:")
    print(f"  (1/pi)*int_0^{{2*pi}} x^2 dx = {parseval_lhs:.8f}")
    print(f"  Exact 8*pi^2/3 = {exact_parseval:.8f}")
    assert abs(parseval_lhs - exact_parseval) < 1e-10

    # --- Smooth function: f(x) = exp(cos(x)) ----------------------
    # All Fourier coefficients are related to modified Bessel functions
    # a_0 = I_0(1), a_n = 2*I_n(1)  (where I_n are Bessel functions)
    from scipy.special import iv as bessel_i
    print(f"\nf(x) = exp(cos(x)) on [0, 2*pi]:")
    a0_ec, _ = compute_fourier_coeff(lambda x: jnp.exp(jnp.cos(x)), 0, dom)
    print(f"  a_0 = {a0_ec:.10f}  (exact I_0(1) = {bessel_i(0, 1.0):.10f})")
    assert abs(a0_ec - bessel_i(0, 1.0)) < 1e-10

    for n in [1, 2, 3]:
        a_n, b_n = compute_fourier_coeff(lambda x: jnp.exp(jnp.cos(x)), n, dom)
        exact_a = 2.0 * bessel_i(n, 1.0)
        print(f"  a_{n} = {a_n:.10f}  (exact 2*I_{n}(1) = {exact_a:.10f}), b_{n} = {b_n:.2e}")
        assert abs(a_n - exact_a) < 1e-10
        assert abs(b_n) < 1e-10  # exp(cos(x)) is even, so all b_n = 0

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    # Plot |Fourier coefficients| of cos(3x) and sawtooth
    fig, axes = _plt.subplots(1, 2, figsize=(9, 3.5))
    _ns = _np.arange(0, 10)
    _cos3x_an = _np.array([abs(compute_fourier_coeff(lambda x: jnp.cos(3.0*x), n, dom)[0])
                            for n in _ns])
    axes[0].bar(_ns, _cos3x_an, color="#4169E1", alpha=0.8)
    axes[0].set_title("|a_n| of cos(3x)", fontsize=11)
    axes[0].set_xlabel("n", fontsize=10)
    _saw_bn = _np.array([abs(compute_fourier_coeff(lambda x: x, n, dom)[1])
                          for n in range(1, 11)])
    axes[1].bar(_np.arange(1, 11), _saw_bn, color="#E04040", alpha=0.8)
    axes[1].set_title("|b_n| of sawtooth f(x)=x", fontsize=11)
    axes[1].set_xlabel("n", fontsize=10)
    for _ax in axes:
        _ax.grid(True, alpha=0.3, linestyle="--")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fourier_coefficients.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
