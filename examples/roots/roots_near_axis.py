"""Complex roots near the real axis.

A chebfun is an analytic function inside a Bernstein ellipse; this gives
access to complex roots near the real interval. Based on Chebfun example
roots/RootsNearAxis.m by Nick Trefethen (October 2011).

Original: https://www.chebfun.org/examples/roots/RootsNearAxis.html
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
                          '../../docs/images/roots')
    os.makedirs(outdir, exist_ok=True)

    # f = 3 + sin(x) + sin(pi*x) on [0, 30]
    # This function has no real roots but has complex roots near the axis.
    f = cj.chebfun(lambda x: 3 + jnp.sin(x) + jnp.sin(jnp.pi * x),
                   domain=(0.0, 30.0))

    # Real roots: should be none
    real_roots = f.roots()
    print(f"Real roots of f: {len(real_roots)} (expected: 0)")

    # Polynomial degree
    degree = len(f) - 1
    print(f"Polynomial degree: {degree}")

    # --- Plot function and illustrate the concept -----------------------
    fig, axes = plt.subplots(1, 2)

    xx = np.linspace(0, 30, 600)
    fv = np.array(f(jnp.array(xx)))
    axes[0].plot(xx, fv, 'b-', linewidth=1.5)
    axes[0].axhline(0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_title('$f(x) = 3 + \sin(x) + \sin(\pi x)$', fontsize=11)
    axes[0].set_xlabel('$x$')
    axes[0].set_ylabel('$f(x)$')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-1.5, 5.5)

    # Bernstein ellipse for f
    # The Chebfun "ellipse" for a function of degree n on [a,b] is the
    # largest ellipse in the complex plane with semi-axes summing to rho
    # where the function is analytic.
    # For f = 3 + sin(x) + sin(pi*x), the singularities are at complex
    # distance related to 1/pi and 1 from the real axis.
    # We illustrate the ellipse analytically.
    a, b = 0.0, 30.0
    # Map [a,b] to [-1,1]: t = (2x - (a+b)) / (b-a)
    # Bernstein ellipse in the t-variable with parameter rho
    theta = np.linspace(0, 2 * np.pi, 400)
    # For illustration, show a Bernstein ellipse with rho = 1.15
    for rho in [1.05, 1.10, 1.20]:
        t_ellipse = (rho * np.exp(1j * theta) + (1 / rho) * np.exp(-1j * theta)) / 2
        # Map back to [a,b]
        x_ellipse = ((b - a) * t_ellipse + (a + b)) / 2
        axes[1].plot(x_ellipse.real, x_ellipse.imag, '-',
                     linewidth=0.9, alpha=0.7, label=rf'$\rho={rho}$')
    axes[1].plot(xx, np.zeros_like(xx), 'k-', linewidth=1.5, label='real axis')
    axes[1].set_aspect('equal')
    axes[1].set_title("Bernstein ellipses for $f$ on $[0, 30]$", fontsize=11)
    axes[1].set_xlabel('Re($z$)')
    axes[1].set_ylabel('Im($z$)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-5, 5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'roots_near_axis.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Minimum value of f (should be > 0, confirming no real roots)
    _, min_val = f.min()
    print(f"min(f) = {min_val:.6f}  (> 0 confirms no real roots)")
    assert min_val > 0, f"f has real roots (min={min_val})"

    print("roots_near_axis: done")
    return True


if __name__ == "__main__":
    run()
