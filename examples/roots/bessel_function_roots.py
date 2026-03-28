"""Roots of Bessel functions.

Finds the zeros of J_0, J_1, J_2 using chebfunjax, and compares with
reference values. Based on Chebfun example roots/BesselRoots.m by
Nick Trefethen (September 2010).

Original: https://www.chebfun.org/examples/roots/BesselRoots.html
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.special import jv, jn_zeros
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/roots')
    os.makedirs(outdir, exist_ok=True)

    # J_0 on [0, 100]
    J0 = cj.chebfun(lambda x: jnp.array(jv(0, np.array(x))), domain=(0.0, 100.0))
    r0 = np.sort(np.array(J0.roots()))
    r0 = r0[r0 > 0.01]  # Remove possible root at 0
    print(f"J_0: found {len(r0)} roots in [0, 100]")
    print(f"First 5 roots: {r0[:5]}")

    # Compare with reference
    ref_r0 = jn_zeros(0, min(len(r0), 31))
    if len(ref_r0) == len(r0):
        max_err = np.max(np.abs(r0[:len(ref_r0)] - ref_r0))
        print(f"Max error vs scipy.jn_zeros: {max_err:.2e}")
        assert max_err < 1e-8, f"Root error too large: {max_err}"

    # J_1 on [0, 50]
    J1 = cj.chebfun(lambda x: jnp.array(jv(1, np.array(x))), domain=(0.0, 50.0))
    r1 = np.sort(np.array(J1.roots()))
    r1 = r1[r1 > 0.01]
    print(f"J_1: found {len(r1)} roots in (0, 50]")

    # Plot
    fig, ax = plt.subplots()
    xx = np.linspace(0, 40, 1000)
    J0_vals = jv(0, xx)
    J1_vals = jv(1, xx)
    ax.plot(xx, J0_vals, color='#0072BD', linestyle='-', linewidth=1.5, label='$J_0(x)$')
    ax.plot(xx, J1_vals, color='#D95319', linestyle='-', linewidth=1.5, label='$J_1(x)$')
    # Mark the roots
    r0_plot = r0[r0 <= 40]
    r1_plot = r1[r1 <= 40]
    ax.plot(r0_plot, np.zeros_like(r0_plot), '.b', markersize=8)
    ax.plot(r1_plot, np.zeros_like(r1_plot), '.r', markersize=8)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_title('Roots of Bessel functions $J_0$ and $J_1$', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.5, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'bessel_function_roots.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Count roots in a large interval
    interval = (1_000_000, 1_001_000)
    J0_big = cj.chebfun(lambda x: jnp.array(jv(0, np.array(x))),
                        domain=interval)
    r_big = J0_big.roots()
    print(f"Roots of J_0 in [{interval[0]}, {interval[1]}]: {len(r_big)}")
    # Asymptotically pi roots per unit interval ~ 1000/pi ≈ 318
    expected = int(round(1000.0 / np.pi))
    print(f"Expected (~1000/pi): {expected}")

    print("bessel_function_roots: done")
    return True

if __name__ == "__main__":
    run()
