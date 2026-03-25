"""Inner products and orthogonality of Chebfun functions.

Demonstrates computing L2 inner products using Chebfun integration,
verifying orthogonality of Legendre and Chebyshev polynomials.

Credit: Inspired by Chebfun linalg examples.
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


def run():
    print("=" * 60)
    print("Inner products and orthogonality")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # --- L2 inner product: <f, g> = integral of f*g over [-1,1] ------
    f = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), domain=dom)
    g = cj.chebfun(lambda x: jnp.cos(jnp.pi * x), domain=dom)

    # <sin(pi x), cos(pi x)> = (1/2) int sin(2*pi*x) dx from -1 to 1 = 0
    fg = cj.chebfun(lambda x: jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * x), domain=dom)
    inner_fg = float(fg.sum())
    print(f"\n<sin(pi x), cos(pi x)> = {inner_fg:.2e}  (exact: 0)")
    assert abs(inner_fg) < 1e-12

    # <sin(pi x), sin(pi x)> = int sin^2(pi x) dx from -1 to 1 = 1
    f2 = cj.chebfun(lambda x: jnp.sin(jnp.pi * x)**2, domain=dom)
    norm_f_sq = float(f2.sum())
    print(f"<sin(pi x), sin(pi x)> = {norm_f_sq:.12f}  (exact: 1.0)")
    assert abs(norm_f_sq - 1.0) < 1e-12

    # --- Legendre polynomial orthogonality ---------------------------
    # P_0(x) = 1, P_1(x) = x, P_2(x) = (3x^2-1)/2
    # <P_m, P_n> = 2/(2n+1) * delta_{mn}
    P0 = cj.chebfun(lambda x: jnp.ones_like(x), domain=dom)
    P1 = cj.chebfun(lambda x: x, domain=dom)
    P2 = cj.chebfun(lambda x: (3.0*x**2 - 1.0) / 2.0, domain=dom)

    # Cross terms should vanish
    ip_01 = float(cj.chebfun(lambda x: jnp.ones_like(x) * x, domain=dom).sum())
    ip_02 = float(cj.chebfun(lambda x: (3.0*x**2 - 1.0)/2.0, domain=dom).sum())
    ip_12 = float(cj.chebfun(lambda x: x * (3.0*x**2 - 1.0)/2.0, domain=dom).sum())

    print(f"\nLegendre orthogonality:")
    print(f"  <P0, P1> = {ip_01:.2e}  (exact: 0)")
    print(f"  <P0, P2> = {ip_02:.2e}  (exact: 0)")
    print(f"  <P1, P2> = {ip_12:.2e}  (exact: 0)")
    assert abs(ip_01) < 1e-14
    assert abs(ip_02) < 1e-14
    assert abs(ip_12) < 1e-14

    # Diagonal norms: <Pn, Pn> = 2/(2n+1)
    ip_00 = float(cj.chebfun(lambda x: jnp.ones_like(x)**2, domain=dom).sum())
    ip_11 = float(cj.chebfun(lambda x: x**2, domain=dom).sum())
    ip_22 = float(cj.chebfun(lambda x: ((3.0*x**2 - 1.0)/2.0)**2, domain=dom).sum())
    print(f"  <P0, P0> = {ip_00:.12f}  (exact: 2/1 = 2)")
    print(f"  <P1, P1> = {ip_11:.12f}  (exact: 2/3 = {2/3:.12f})")
    print(f"  <P2, P2> = {ip_22:.12f}  (exact: 2/5 = {2/5:.12f})")
    assert abs(ip_00 - 2.0) < 1e-13
    assert abs(ip_11 - 2.0/3.0) < 1e-13
    assert abs(ip_22 - 2.0/5.0) < 1e-13

    # --- Chebyshev polynomial weighted orthogonality -----------------
    # T_m, T_n orthogonal with weight w(x) = 1/sqrt(1-x^2)
    # <T_m, T_n>_w = int T_m(x)*T_n(x)/sqrt(1-x^2) dx = 0 if m != n
    # T_0(x) = 1, T_1(x) = x, T_2(x) = 2x^2 - 1, T_3(x) = 4x^3 - 3x
    # Odd * even = odd function => int over [-1,1] = 0 (no weight needed)
    # T_1 (odd) * T_2 (even) = odd => int = 0
    # T_0 (even) * T_1 (odd) = odd => int = 0
    ip_T0T1 = float(cj.chebfun(lambda x: 1.0 * x, domain=dom).sum())
    ip_T1T2 = float(cj.chebfun(lambda x: x * (2.0*x**2 - 1.0), domain=dom).sum())
    print(f"\nChebyshev parity orthogonality (odd*even = 0):")
    print(f"  int T0*T1 dx = {ip_T0T1:.2e}  (exact: 0, T1 is odd)")
    print(f"  int T1*T2 dx = {ip_T1T2:.2e}  (exact: 0, T1*T2 is odd)")
    assert abs(ip_T0T1) < 1e-14
    assert abs(ip_T1T2) < 1e-14

    # T_0*T_2 are both even, so not zero without weight
    ip_T0T2 = float(cj.chebfun(lambda x: 1.0 * (2.0*x**2 - 1.0), domain=dom).sum())
    print(f"  int T0*T2 dx = {ip_T0T2:.10f}  (exact: -2/3 = {-2/3:.10f})")
    assert abs(ip_T0T2 - (-2.0/3.0)) < 1e-13

    # --- Norm of a polynomial ----------------------------------------
    # ||x^2 - 1/3||_{L2} = sqrt(int (x^2-1/3)^2 dx from -1 to 1)
    # = sqrt(8/45)
    p = cj.chebfun(lambda x: x**2 - 1.0/3.0, domain=dom)
    norm_p = float(p.norm(2))
    exact_norm = float(jnp.sqrt(jnp.array(8.0/45.0)))
    print(f"\n||x^2 - 1/3||_2 = {norm_p:.12f}")
    print(f"Exact sqrt(8/45) = {exact_norm:.12f}")
    assert abs(norm_p - exact_norm) < 1e-12

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(f, title="Inner products: sin(πx) and cos(πx)",
                   label="sin(πx)")
    plot(g, ax=ax, color="#E04040", label="cos(πx)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "chebfun_inner_products.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
