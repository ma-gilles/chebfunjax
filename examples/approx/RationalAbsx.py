"""Rational approximation of abs(x) with minimax.

Best rational approximation of |x| achieves root-exponential convergence
O(exp(-C*sqrt(n))), far better than polynomial O(1/n).

Credit: Silviu Filip, Yuji Nakatsukasa, and Nick Trefethen, May 2017.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/RationalAbsx.html
"""

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

from chebfunjax.utils.aaa import aaa

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')

def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    xx = np.linspace(-1.0, 1.0, 600)
    abs_x = np.abs(xx)

    # Polynomial errors (converge as O(1/n))
    f = cj.chebfun(jnp.abs)
    poly_ns = [10, 20, 40, 80, 160]
    poly_errs = []
    for n in poly_ns:
        pn = f.polyfit(n)
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xx])
        poly_errs.append(np.max(np.abs(pn_vals - abs_x)))

    # AAA rational errors (converge as O(exp(-C*sqrt(n))))
    xs_aaa = jnp.linspace(-1.0, 1.0, 400)
    ys_aaa = jnp.abs(xs_aaa)
    r_aaa, pol, *_ = aaa(ys_aaa, xs_aaa)
    r_vals = np.array([float(r_aaa(jnp.array(x)).real) for x in xx])
    rat_err_full = np.max(np.abs(r_vals - abs_x))

    # Newman-type bound: 2*exp(-C*sqrt(n))
    C = 3.14159  # approximate Newman constant
    ns_cont = np.linspace(1, 200, 200)
    newman_bound = 2.0 * np.exp(-C * np.sqrt(ns_cont / 2.0))

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.semilogy(poly_ns, poly_errs, color='#0072BD', linestyle='.-', lw=1.5, ms=10, label='poly errors')
    ax.semilogy(poly_ns, 1.0 / np.array(poly_ns), color='#0072BD', linestyle='--', lw=1.2,
                label='O(1/n) reference')
    ax.axhline(rat_err_full, color='#D95319', ls='--', lw=1.5,
               label=f'AAA ({len(pol)} poles, err={rat_err_full:.2e})')
    ax.set_title('Approximation errors for |x|', fontsize=10)
    ax.legend(fontsize=8)
    ax2 = axes[1]
    ax2.semilogy(ns_cont, newman_bound, 'r', lw=1.5, label='exp(-C√n) bound')
    ax2.set_title('Root-exponential convergence rate', fontsize=10)
    ax2.legend(fontsize=9)
    fig.suptitle('Polynomial O(1/n) vs. rational O(exp(-C√n)) for |x|', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'RationalAbsx.png'), dpi=150)
    plt.close(fig)

    print(f"RationalAbsx: poly deg-80 err={poly_errs[-2]:.3e}, "
          f"AAA err={rat_err_full:.3e}")
    return True

if __name__ == '__main__':
    run()
