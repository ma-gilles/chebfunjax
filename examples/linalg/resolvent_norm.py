"""Inner product spaces, norms, and functional analysis via Chebfun.

Demonstrates L2 norms, Rayleigh quotients, and functional analysis
concepts using Chebfun integration.

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
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.plotting import plot

def run():
    print("=" * 60)
    print("Inner product spaces and norms")
    print("=" * 60)

    # --- Norm of a Chebfun: verification of ||u||_2 ----------------
    # ||sin(n*pi*x)||_2 on [0,1] = 1/sqrt(2)
    dom = (0.0, 1.0)
    for n in [1, 2, 5]:
        u_n = cj.chebfun(lambda x, n=n: jnp.sin(n * float(jnp.pi) * x), domain=dom)
        norm_n = float(u_n.norm(2))
        exact_norm = float(1.0 / jnp.sqrt(jnp.array(2.0)))
        print(f"\n  ||sin({n}*pi*x)||_2 on [0,1] = {norm_n:.12f}")
        print(f"  Exact: 1/sqrt(2) = {exact_norm:.12f}")
        assert abs(norm_n - exact_norm) < 1e-11

    # --- Rayleigh quotient: for -d^2/dx^2 and sin(pi*x) on [0,1] ----
    # Rayleigh quotient R(u) = <u', u'> / <u, u>
    # For u = sin(pi*x): R = pi^2
    print(f"\n  Rayleigh quotient test: R(sin(pi*x)) = pi^2")
    u1 = cj.chebfun(lambda x: jnp.sin(float(jnp.pi) * x), domain=dom)
    u1p = u1.diff()
    # <u1', u1'> = int (pi*cos(pi*x))^2 dx from 0 to 1 = pi^2/2
    u1p_sq = cj.chebfun(lambda x: (float(jnp.pi) * jnp.cos(float(jnp.pi) * x))**2,
                         domain=dom)
    numerator = float(u1p_sq.sum())
    # <u1, u1> = int sin^2(pi*x) dx = 1/2
    u1_sq = cj.chebfun(lambda x: jnp.sin(float(jnp.pi) * x)**2, domain=dom)
    denominator = float(u1_sq.sum())
    rayleigh = numerator / denominator
    exact_rayleigh = float(jnp.pi**2)
    print(f"  R = {rayleigh:.10f}")
    print(f"  Exact pi^2 = {exact_rayleigh:.10f}")
    assert abs(rayleigh - exact_rayleigh) < 1e-8

    # --- BVP solution norm: -u'' = f on [0,1], u(0)=u(1)=0 ----------
    from chebfunjax.operators.chebop import Chebop

    # f = pi^2 * sin(pi*x); exact solution u = sin(pi*x)
    pi = float(jnp.pi)
    N = Chebop(lambda x, u: -u.diff(2), domain=(0.0, 1.0))
    N.lbc = 0.0
    N.rbc = 0.0
    f_rhs = cj.chebfun(lambda x: pi**2 * jnp.sin(pi * x), domain=(0.0, 1.0))
    u_sol = N.solve(f_rhs)
    x_test = jnp.linspace(0.0, 1.0, 200)
    exact_u = jnp.sin(pi * x_test)
    err_u = float(jnp.max(jnp.abs(u_sol(x_test) - exact_u)))
    print(f"\n  -u'' = pi^2*sin(pi*x): ||u - sin(pi*x)||_inf = {err_u:.2e}")
    assert err_u < 1e-8

    # Verify norm: ||u_sol||_2 = ||sin(pi*x)||_2 = 1/sqrt(2)
    norm_sol = float(u_sol.norm(2))
    print(f"  ||u_sol||_2 = {norm_sol:.12f}  (exact: 1/sqrt(2) = {1/np.sqrt(2):.12f})")
    assert abs(norm_sol - 1.0/np.sqrt(2.0)) < 1e-8

    # --- Poincare inequality: ||u||_2 <= (1/pi) * ||u'||_2 ----------
    # For u = sin(pi*x) on [0,1]: ||u||_2 = 1/sqrt(2), ||u'||_2 = pi/sqrt(2)
    # ||u||_2 / ||u'||_2 = 1/pi (tight constant for Poincare)
    u1_norm = float(u1.norm(2))
    u1p_norm = float(u1p.norm(2))
    ratio = u1_norm / u1p_norm
    print(f"\n  Poincare ratio ||u||_2 / ||u'||_2 = {ratio:.12f}")
    print(f"  Exact 1/pi = {1.0/pi:.12f}")
    assert abs(ratio - 1.0/pi) < 1e-10

    # --- Orthogonal complement: projection onto subspace -------------
    # Project sin(x) onto span{cos(x)} on [0, pi]
    dom2 = (0.0, pi)
    f_sin = cj.chebfun(lambda x: jnp.sin(x), domain=dom2)
    g_cos = cj.chebfun(lambda x: jnp.cos(x), domain=dom2)
    # Gram-Schmidt: proj_{g} f = <f, g> / <g, g> * g
    fg = cj.chebfun(lambda x: jnp.sin(x) * jnp.cos(x), domain=dom2)
    gg = cj.chebfun(lambda x: jnp.cos(x)**2, domain=dom2)
    coeff = float(fg.sum()) / float(gg.sum())
    # sin(x) - coeff*cos(x) should be orthogonal to cos(x)
    residual = cj.chebfun(lambda x: jnp.sin(x) - coeff * jnp.cos(x), domain=dom2)
    ip_res_cos = float(cj.chebfun(
        lambda x: (jnp.sin(x) - coeff * jnp.cos(x)) * jnp.cos(x), domain=dom2
    ).sum())
    print(f"\n  Gram-Schmidt: coeff = {coeff:.6f}")
    print(f"  <residual, cos> = {ip_res_cos:.2e}  (should be 0)")
    assert abs(ip_res_cos) < 1e-12

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(u1, title="Eigenfunctions of −d²/dx² on [0,1]",
                   label="sin(πx)")
    plot(u1p, ax=ax, color="#E04040", label="π·cos(πx)")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "resolvent_norm.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
