"""The Gamma function and its poles.

Explores the gamma function using chebfunjax, including its poles at
non-positive integers. Based on Chebfun example approx/GammaFun.m
by Nick Hale (December 2009).

Original: https://www.chebfun.org/examples/approx/GammaFun.html
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma as scipy_gamma
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/approx')
    os.makedirs(outdir, exist_ok=True)

    # The gamma function on subintervals avoiding poles
    # Poles at 0, -1, -2, -3, -4
    xx = np.linspace(0.1, 4.0, 800)
    fig, ax = plt.subplots()

    # Plot on (0,4] — analytic piece
    f_pos = cj.chebfun(lambda x: jnp.array(scipy_gamma(np.array(x))),
                       domain=(0.1, 4.0))
    xx_pos = np.linspace(0.15, 4.0, 600)
    ax.plot(xx_pos, np.array(f_pos(jnp.array(xx_pos))), 'b-', linewidth=1.8,
            label=r'$\Gamma(x)$')

    # On (-1, 0) and (-2, -1) and (-3, -2)
    for a, b in [(-0.95, -0.05), (-1.95, -1.05), (-2.95, -2.05), (-3.95, -3.05)]:
        f_piece = cj.chebfun(lambda x, a=a, b=b: jnp.array(scipy_gamma(np.array(x))),
                             domain=(a, b))
        xp = np.linspace(a + 0.01, b - 0.01, 200)
        vals = np.array(f_piece(jnp.array(xp)))
        ax.plot(xp, np.clip(vals, -10, 10), 'b-', linewidth=1.8)

    ax.axhline(0, color='k', linewidth=0.7)
    for pole in [0, -1, -2, -3, -4]:
        ax.axvline(pole, color='gray', linewidth=0.5, linestyle='--', alpha=0.6)
    ax.set_ylim(-8, 8)
    ax.set_xlim(-4.2, 4.2)
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_title(r'The Gamma function $\Gamma(x)$ on $[-4, 4]$', fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'gamma_function.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Verify some values
    f_check = cj.chebfun(lambda x: jnp.array(scipy_gamma(np.array(x))),
                         domain=(0.5, 3.5))
    val_1 = float(f_check(jnp.array(1.0)))   # Gamma(1) = 1
    val_2 = float(f_check(jnp.array(2.0)))   # Gamma(2) = 1
    val_3 = float(f_check(jnp.array(3.0)))   # Gamma(3) = 2
    val_half = float(f_check(jnp.array(0.5)))  # Gamma(0.5) = sqrt(pi)
    print(f"Gamma(1)   = {val_1:.12f}  (exact: 1.0)")
    print(f"Gamma(2)   = {val_2:.12f}  (exact: 1.0)")
    print(f"Gamma(3)   = {val_3:.12f}  (exact: 2.0)")
    print(f"Gamma(0.5) = {val_half:.12f}  (exact: sqrt(pi)={np.sqrt(np.pi):.12f})")
    assert abs(val_1 - 1.0) < 1e-10
    assert abs(val_2 - 1.0) < 1e-10
    assert abs(val_3 - 2.0) < 1e-10
    assert abs(val_half - np.sqrt(np.pi)) < 1e-10

    print("gamma_function: done")
    return True


if __name__ == "__main__":
    run()
