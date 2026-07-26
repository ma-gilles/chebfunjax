"""Chebyshev polynomials T_n and their properties.

Explores the Chebyshev polynomials T_0, T_1, ..., T_10, including their
three-term recurrence, extrema, and orthogonality.

Original: https://www.chebfun.org/examples/cheb/
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()



def chebyshev_T(n, x):
    """Chebyshev polynomial T_n via cosine formula."""
    return np.cos(n * np.arccos(np.clip(x, -1, 1)))


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/cheb')
    os.makedirs(outdir, exist_ok=True)

    # --- Plot T_0 through T_6 as a single overlay -----------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    xx = np.linspace(-1, 1, 500)

    # matplotlib default (tab10) color cycle, to match the reference render
    cyc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2']
    for n in range(7):
        Tn = chebyshev_T(n, xx)
        ax.plot(xx, Tn, color=cyc[n], linewidth=1.5, label=f'T_{n}')

    ax.set_title('Chebyshev polynomials T_0, ..., T_6', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('T_n(x)')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'chebyshev_polynomials.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Orthogonality check using chebfunjax ----------------------------
    # T_m and T_n are orthogonal w.r.t. weight w = 1/sqrt(1-x^2)
    # <T_m, T_n>_w = integral_{-1}^{1} T_m(x) T_n(x) / sqrt(1-x^2) dx
    # = 0 if m != n, pi/2 if m=n>0, pi if m=n=0
    print("Orthogonality check:")
    for m in range(4):
        for n in range(4):
            # Use chebfun on a slightly restricted domain to avoid endpoint singularity
            dom = (-0.9999, 0.9999)
            Tm = cj.chebfun(lambda x, m=m: jnp.array(chebyshev_T(m, np.array(x))),
                            domain=dom)
            Tn = cj.chebfun(lambda x, n=n: jnp.array(chebyshev_T(n, np.array(x))),
                            domain=dom)
            w = cj.chebfun(lambda x: 1.0 / jnp.sqrt(1.0 - x**2), domain=dom)
            inn = float((w * Tm * Tn).sum())
            expected = 0.0 if m != n else (np.pi if m == 0 else np.pi / 2)
            print(f"  <T_{m}, T_{n}>_w = {inn:8.4f}  (expected: {expected:.4f})")

    # --- Three-term recurrence ------------------------------------------
    # T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
    x_test = np.array([0.3, 0.7, -0.5])
    print("\nThree-term recurrence check at x = 0.3, 0.7, -0.5:")
    for xt in x_test:
        for n in range(2, 8):
            Tn_rec = 2 * xt * chebyshev_T(n, xt) - chebyshev_T(n - 1, xt)
            Tn_direct = chebyshev_T(n + 1, xt)
            assert abs(Tn_rec - Tn_direct) < 1e-12, f"Recurrence failed at n={n}, x={xt}"
    print("  All recurrences verified.")

    # --- Minimax property -----------------------------------------------
    # T_n / 2^{n-1} is the monic polynomial of degree n with smallest infinity norm
    print("\nMinimax property (inf norm of T_n = 1 for all n):")
    for n in range(1, 8):
        inf_norm = np.max(np.abs(chebyshev_T(n, xx)))
        print(f"  ||T_{n}||_inf = {inf_norm:.10f}  (should be 1.0)")
        assert abs(inf_norm - 1.0) < 1e-10

    print("chebyshev_polynomials: done")
    return True


if __name__ == "__main__":
    run()
