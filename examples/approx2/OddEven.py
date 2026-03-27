"""Odd and even best approximations.

Demonstrates that the best polynomial approximation to an odd function
is an odd polynomial, and to an even function an even polynomial.

Credit: Mohsin Javed and Nick Trefethen, March 2015.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/OddEven.html
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


_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Even function: |x|
    f_even = cj.chebfun(jnp.abs)
    # Odd function: sign(x) (on piecewise domain)
    f_odd = cj.chebfun(lambda x: jnp.sign(x), domain=(-1.0, 0.0, 1.0))

    # Best L2 polynomial approximations
    p_even_10 = f_even.polyfit(10)
    p_odd_9 = f_odd.polyfit(9)

    xx = np.linspace(-1.0, 1.0, 400)
    even_vals = np.abs(xx)
    odd_vals = np.sign(xx)

    p_e10_vals = np.array([float(p_even_10(jnp.array(x))) for x in xx])
    p_o9_vals = np.array([float(p_odd_9(jnp.array(x))) for x in xx])

    err_even = np.abs(p_e10_vals - even_vals)
    err_odd = np.abs(p_o9_vals - odd_vals)

    # Check coefficients: even approximant should have only even-degree terms
    c_even = np.array(p_even_10.coeffs)
    c_odd = np.array(p_odd_9.coeffs)
    # Chebyshev coefficients: even degree T_k for even function
    odd_coeff_norm = np.sum(np.abs(c_even[1::2]))  # should be ~0
    even_coeff_norm = np.sum(np.abs(c_odd[::2]))   # should be ~0

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(xx, even_vals, 'b', lw=1.8, label='|x|')
    axes[0, 0].plot(xx, p_e10_vals, 'r--', lw=1.5, label='poly deg 10')
    axes[0, 0].set_title('Even function |x| and even approx', fontsize=10)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(xx, odd_vals, 'b', lw=1.8, label='sign(x)')
    axes[0, 1].plot(xx, p_o9_vals, 'r--', lw=1.5, label='poly deg 9')
    axes[0, 1].set_title('Odd function sign(x) and odd approx', fontsize=10)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(np.arange(len(c_even)), np.abs(c_even) + 1e-18,
                        'b.', ms=7)
    axes[1, 0].set_title(f'Even polynomial coefficients (odd-norm={odd_coeff_norm:.1e})', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(np.arange(len(c_odd)), np.abs(c_odd) + 1e-18,
                        'r.', ms=7)
    axes[1, 1].set_title(f'Odd polynomial coefficients (even-norm={even_coeff_norm:.1e})', fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('x' if ax in axes[:1, :].flat else 'degree')

    fig.suptitle('Odd and even best polynomial approximations', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'OddEven.png'), dpi=150)
    plt.close(fig)

    print(f"OddEven: even poly odd-coeff norm={odd_coeff_norm:.2e}, "
          f"odd poly even-coeff norm={even_coeff_norm:.2e}")
    return True


if __name__ == '__main__':
    run()
