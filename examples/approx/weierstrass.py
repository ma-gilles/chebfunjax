"""A pathological function of Weierstrass.

The Weierstrass function is everywhere continuous but nowhere differentiable.
Based on Chebfun example approx/WeierstrassFunction.m by Hrothgar (October 2013).

Original: https://www.chebfun.org/examples/approx/WeierstrassFunction.html
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
                          '../../docs/images/approx')
    os.makedirs(outdir, exist_ok=True)

    # F(x) = sum_{k=0}^{inf} 2^{-k} cos(pi/2 * 4^k * x)  on [-1,1]
    # We compute 8 partial sums and plot the last one.
    terms = []
    for k in range(9):
        terms.append(cj.chebfun(lambda x, k=k: 2.0**(-k) * jnp.cos(jnp.pi / 2 * 4**k * x)))

    F = terms[0]
    partial = [F]
    for k in range(1, 8):
        F = F + terms[k]
        partial.append(F)

    # The exact integral of F over [-1,1] = 4/pi (only first term contributes)
    exact_integral = 4.0 / float(jnp.pi)
    approx_integral = float(partial[-1].sum())
    print(f"Integral of F_8 = {approx_integral:.10f}  (exact: {exact_integral:.10f})")
    print(f"Error: {abs(approx_integral - exact_integral):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Full function on [-1, 1]
    xx = np.linspace(-1.0, 1.0, 2000)
    F_vals = np.array(partial[-1](jnp.array(xx)))
    axes[0].plot(xx, F_vals, 'k-', linewidth=0.8)
    axes[0].set_title(r'Weierstrass function $\sum_{k=0}^{8} 2^{-k}\cos(\frac{\pi}{2}4^k x)$',
                      fontsize=10)
    axes[0].set_xlabel('$x$')
    axes[0].grid(True, alpha=0.3)

    # Zoomed in to show the fractal-like structure
    xx_zoom = np.linspace(0.0, 0.005, 1000)
    F_zoom = np.array(partial[-1](jnp.array(xx_zoom)))
    axes[1].plot(xx_zoom, F_zoom, 'b-', linewidth=0.9)
    axes[1].set_title('Close-up: $x \in [0, 0.005]$ — fractal structure', fontsize=10)
    axes[1].set_xlabel('$x$')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'weierstrass.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("weierstrass: done")
    return True


if __name__ == "__main__":
    run()
