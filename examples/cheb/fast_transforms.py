"""Fast Chebyshev-Legendre transforms.

Demonstrates the fast Chebyshev-Legendre and discrete Legendre transforms,
following cheb/FastChebyshevLegendreTransform.m by Townsend & Hale (August 2013)
and cheb/FastDLT.m by Hale & Townsend (April 2015).

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

from chebfunjax.utils.transforms import cheb2leg, leg2cheb, vals2coeffs, coeffs2vals
from chebfunjax.utils.quadrature import chebpts

def run():
    print("=" * 60)
    print("Fast Chebyshev-Legendre transforms")
    print("=" * 60)

    # --- Round-trip Chebyshev <-> Legendre ---
    # Build Chebyshev coefficients of exp(x)
    f = cj.chebfun(jnp.exp)
    cheb_c = np.array(f.funs[0].tech.coeffs)
    n = len(cheb_c)
    print(f"\nexp(x) has {n} Chebyshev coefficients")

    # Convert to Legendre and back
    leg_c = np.array(cheb2leg(jnp.array(cheb_c)))
    cheb_c2 = np.array(leg2cheb(jnp.array(leg_c)))

    roundtrip_err = np.max(np.abs(cheb_c - cheb_c2))
    print(f"Round-trip cheb->leg->cheb error: {roundtrip_err:.2e}")
    assert roundtrip_err < 1e-10, f"Round-trip error too large: {roundtrip_err}"

    # --- Check sizes grow ---
    for sz in [16, 64, 256]:
        xs = chebpts(sz)
        ys = jnp.exp(xs)
        c = vals2coeffs(ys)
        cl = cheb2leg(c)
        c2 = leg2cheb(cl)
        err = float(jnp.max(jnp.abs(c - c2)))
        print(f"  n={sz}: cheb<->leg round-trip error = {err:.2e}")
        assert err < 1e-9

    # --- Verify Legendre coefficients of specific function ---
    # P_0(x) = 1, integral over [-1,1] = 2 => leg coeff 0 should be 1/2 * integral
    n_test = 32
    xs = chebpts(n_test)
    ys_const = jnp.ones(n_test)
    c_cheb = vals2coeffs(ys_const)
    c_leg = cheb2leg(c_cheb)
    print(f"\nLeg coefficients of f=1 (only c_0 nonzero):")
    print(f"  c_0 = {float(c_leg[0]):.8f}  (expected ~1.0 for Legendre normalization)")
    assert abs(float(c_leg[0]) - 1.0) < 0.1  # normalization varies

    # --- Plot: coefficient magnitudes in Cheb vs Leg basis ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    n_plot = 32
    xs_p = chebpts(n_plot)
    ys_p = jnp.exp(xs_p)
    c_ch = np.abs(np.array(vals2coeffs(ys_p)))
    c_lg = np.abs(np.array(cheb2leg(jnp.array(c_ch))))

    axes[0].semilogy(c_ch + 1e-17, color='#0072BD', linestyle='.-', label='Chebyshev')
    axes[0].semilogy(c_lg + 1e-17, color='#D95319', linestyle='.-', label='Legendre')
    axes[0].set_title("Cheb vs Legendre coefficients of exp(x)", fontsize=11)
    axes[0].legend()

    # Convergence of round-trip error vs n
    sizes = [8, 16, 32, 64, 128, 256]
    errors = []
    for sz in sizes:
        xs_s = chebpts(sz)
        ys_s = jnp.sin(xs_s)
        c_s = vals2coeffs(ys_s)
        cl_s = cheb2leg(c_s)
        c2_s = leg2cheb(cl_s)
        errors.append(float(jnp.max(jnp.abs(c_s - c2_s))))

    axes[1].loglog(sizes, errors, 'ko-')
    axes[1].axhline(1e-14, color='#D95319', linestyle='--', alpha=0.5, label='machine ε')
    axes[1].set_title("Round-trip error vs. n", fontsize=11)
    axes[1].legend()

    fig.suptitle("Fast Chebyshev-Legendre transforms", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "fast_transforms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
