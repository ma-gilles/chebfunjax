"""Rational approximation of monomials.

The monomial x^200 on [0,1] is hard for polynomials but very efficiently
approximated by rational functions of low type.

Credit: Yuji Nakatsukasa and Nick Trefethen, May 2019.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/Rationalxn.html
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

    # x^200 on [0, 1]
    exp_val = 200
    f = cj.chebfun(lambda x: x**exp_val, domain=(0.0, 1.0))
    xs = np.linspace(0.0, 1.0, 500)
    true_vals = xs**exp_val

    # AAA rational approximation
    xs_aaa = jnp.linspace(0.0, 1.0, 300)
    ys_aaa = xs_aaa**exp_val
    r_aaa, pol, *_ = aaa(ys_aaa, xs_aaa)
    r_vals = np.array([float(r_aaa(jnp.array(x)).real) for x in xs])
    err_aaa = np.abs(r_vals - true_vals)

    # Polynomial approximation
    poly_ns = [10, 50, 100]
    poly_errs = []
    for n in poly_ns:
        pn = f.polyfit(n)
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xs])
        poly_errs.append(np.max(np.abs(pn_vals - true_vals)))

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.plot(xs, true_vals, 'b', lw=1.8, label=f'x^{exp_val}')
    ax.plot(xs, r_vals, color='#D95319', linestyle='--', lw=1.5,
            label=f'AAA ({len(pol)} poles, err={np.max(err_aaa):.2e})')
    ax.set_title(f'x^{{{exp_val}}} on [0, 1]', fontsize=11)
    ax.legend(fontsize=9)
    ax2 = axes[1]
    ax2.semilogy(poly_ns, poly_errs, color='#0072BD', linestyle='.-', lw=1.5, ms=10, label='poly')
    ax2.axhline(np.max(err_aaa), color='#D95319', ls='--', lw=1.5,
                label=f'AAA ({len(pol)} poles)')
    ax2.set_title(f'Errors for x^{{{exp_val}}} approximation', fontsize=10)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'Rationalxn.png'), dpi=150)
    plt.close(fig)

    print(f"Rationalxn: x^{exp_val}: poly-100 err={poly_errs[-1]:.3e}, "
          f"AAA err={np.max(err_aaa):.3e}")
    return True

if __name__ == '__main__':
    run()
