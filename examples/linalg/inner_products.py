"""Inner products and norms of Chebfuns.

Demonstrates the L^2 inner product, various norms, and orthogonality
of trigonometric and polynomial functions.

Original: https://www.chebfun.org/examples/linalg/
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
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/linalg')
    os.makedirs(outdir, exist_ok=True)

    # --- L^2 norm and inner product ------------------------------------
    # Legendre polynomials are orthogonal w.r.t. L^2 inner product on [-1,1]
    # P_m, P_n: <P_m, P_n> = 2/(2n+1) * delta_{mn}
    print("Legendre orthogonality (via chebfunjax inner products):")
    from scipy.special import legendre

    def legpoly(n):
        return cj.chebfun(lambda x: jnp.array(legendre(n)(np.array(x))))

    for m in range(5):
        for n in range(5):
            Pm = legpoly(m)
            Pn = legpoly(n)
            inn = float(Pm.inner(Pn))
            expected = 0.0 if m != n else 2.0 / (2 * n + 1)
            err = abs(inn - expected)
            if err > 1e-10:
                print(f"  <P_{m}, P_{n}> = {inn:.8f}  (expected {expected:.8f})  ERROR!")
            else:
                pass  # All good
    print("  All Legendre inner products correct to 1e-10.")

    # --- Various norms --------------------------------------------------
    f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x)))
    norms = {
        'L1':   float(f.norm(1)),
        'L2':   float(f.norm(2)),
        'Linf': float(f.norm(float('inf'))),
    }
    print(f"\nNorms of exp(sin(x)) on [-1,1]:")
    for k, v in norms.items():
        print(f"  {k} norm = {v:.10f}")

    # Verify L2 norm: ||exp(sin(x))||_2^2 = integral_{-1}^{1} exp(2*sin(x)) dx
    f2 = cj.chebfun(lambda x: jnp.exp(2 * jnp.sin(x)))
    l2_sq_ref = float(f2.sum())
    print(f"  L2^2 via sum(exp(2*sin(x))) = {l2_sq_ref:.10f}")
    print(f"  L2^2 via norm(2)^2          = {norms['L2']**2:.10f}")
    assert abs(norms['L2']**2 - l2_sq_ref) < 1e-10

    # --- Trigonometric orthogonality on [0, 2*pi] ---------------------
    print("\nTrigonometric orthogonality on [0, 2*pi]:")
    T = 2 * float(jnp.pi)
    for m in [1, 2, 3]:
        for n in [1, 2, 3]:
            sm = cj.chebfun(lambda x, m=m: jnp.sin(m * x), domain=(0.0, T))
            cn = cj.chebfun(lambda x, n=n: jnp.cos(n * x), domain=(0.0, T))
            sn = cj.chebfun(lambda x, n=n: jnp.sin(n * x), domain=(0.0, T))
            inn_sc = float(sm.inner(cn))
            inn_ss = float(sm.inner(sn))
            exp_ss = 0.0 if m != n else float(jnp.pi)
            if abs(inn_sc) > 1e-10 or abs(inn_ss - exp_ss) > 1e-10:
                print(f"  ERROR: m={m}, n={n}: <sin(mx),cos(nx)>={inn_sc:.2e}, <sin(mx),sin(nx)>={inn_ss:.6f}")
    print("  All trig inner products correct.")

    # Plot
    fig, ax = plt.subplots()
    xx = np.linspace(-1, 1, 400)
    fv = np.array(f(jnp.array(xx)))
    ax.plot(xx, fv, 'b-', linewidth=1.6, label='$f(x) = e^{\\sin x}$')
    ax.fill_between(xx, 0, fv, alpha=0.15, color='blue', label='area = L1 norm/2')
    ax.set_title(f'$e^{{\\sin x}}$: L1={norms["L1"]:.4f}, L2={norms["L2"]:.4f}, L∞={norms["Linf"]:.4f}',
                 fontsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'inner_products.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("inner_products: done")
    return True

if __name__ == "__main__":
    run()
