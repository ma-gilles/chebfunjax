"""Chebyshev polynomials T_n and their properties.

Explores the Chebyshev polynomials T_0, T_1, ..., T_10, including their
three-term recurrence, extrema, and orthogonality.

Original: https://www.chebfun.org/examples/cheb/
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


def chebyshev_T(n, x):
    """Chebyshev polynomial T_n via cosine formula."""
    return np.cos(n * np.arccos(np.clip(x, -1, 1)))


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/cheb')
    os.makedirs(outdir, exist_ok=True)

    # --- Plot T_0 through T_8 -------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    xx = np.linspace(-1, 1, 500)

    for n in range(9):
        ax = axes[n // 3, n % 3]
        Tn = chebyshev_T(n, xx)
        ax.plot(xx, Tn, 'b-', linewidth=1.6)
        ax.axhline(0, color='k', linewidth=0.5)
        # Chebyshev roots: cos((2k-1)*pi/(2n)) for k=1..n
        if n > 0:
            k = np.arange(1, n + 1)
            roots = np.cos((2 * k - 1) * np.pi / (2 * n))
            ax.plot(roots, np.zeros_like(roots), '.r', markersize=6)
        ax.set_title(f'$T_{n}(x)$', fontsize=11)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xlim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Chebyshev polynomials $T_0$ through $T_8$', fontsize=13)
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
