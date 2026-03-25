"""Approximating the square root by polynomials and rational functions.

Compares polynomial and rational approximation of sqrt(x) on [0,1],
illustrating the superior rate of rational approximation near singularities.

Credit: Yuji Nakatsukasa, May 2019.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/MinimaxSqrt.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.utils.aaa import aaa

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # sqrt(x) on [delta, 1] to avoid singularity at 0
    delta = 1e-4
    def sqrt_func(x): return jnp.sqrt(x)
    f = cj.chebfun(sqrt_func, domain=(delta, 1.0))

    xx = np.linspace(delta, 1.0, 500)
    f_true = np.sqrt(xx)

    # Polynomial approximations
    poly_errs = []
    degrees = [5, 10, 20, 40]
    for n in degrees:
        pn = f.polyfit(n)
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xx])
        poly_errs.append(np.max(np.abs(pn_vals - f_true)))

    # AAA rational approximation
    xs_aaa = jnp.linspace(delta, 1.0, 300)
    ys_aaa = jnp.sqrt(xs_aaa)
    r_aaa, pol_aaa, *_ = aaa(ys_aaa, xs_aaa)
    r_vals = np.array([float(r_aaa(jnp.array(x)).real) for x in xx])
    rational_err = np.max(np.abs(r_vals - f_true))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.semilogy(degrees, poly_errs, 'b.-', lw=1.5, ms=10, label='poly L2')
    ax.axhline(rational_err, color='r', lw=1.5, ls='--',
               label=f'AAA rational ({len(pol_aaa)} poles)')
    ax.set_title('Approximation errors for √x on [10⁻⁴, 1]', fontsize=10)
    ax.set_xlabel('polynomial degree n')
    ax.set_ylabel('max error')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    p20 = f.polyfit(20)
    p20_vals = np.array([float(p20(jnp.array(x))) for x in xx])
    err_poly = np.abs(p20_vals - f_true)
    err_rat = np.abs(r_vals - f_true)

    ax2.semilogy(xx, err_poly + 1e-18, 'b', lw=1.5, label='poly deg 20')
    ax2.semilogy(xx, err_rat + 1e-18, 'r', lw=1.5,
                 label=f'AAA rational')
    ax2.set_title('Error curves near singularity at 0', fontsize=10)
    ax2.set_xlabel('x')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Polynomial vs. rational approximation of √x', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'MinimaxSqrt.png'), dpi=150)
    plt.close(fig)

    print(f"MinimaxSqrt: poly deg-20 err={poly_errs[2]:.2e}, "
          f"AAA err={rational_err:.2e}")
    return True


if __name__ == '__main__':
    run()
