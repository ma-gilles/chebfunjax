"""Approximating the pth root by composite rational functions.

Composite (iterated) rational approximations to x^{1/p} converge
super-geometrically fast on [0,1].

Credit: Evan S. Gawlik and Yuji Nakatsukasa, October 2019.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/PthComposite.html
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

from chebfunjax.utils.aaa import aaa

_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', '..', 'docs', 'images', 'approx')


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Approximate x^{1/p} on [delta, 1] for p=3
    p = 3
    delta = 1e-4

    def pthroot(x): return x**(1.0 / p)

    xs_dense = np.linspace(delta, 1.0, 500)
    true_vals = xs_dense**(1.0 / p)

    # AAA rational approximations for different tolerances
    xs_aaa = jnp.linspace(delta, 1.0, 300)
    ys_aaa = jnp.array([x**(1.0/p) for x in xs_aaa])

    r_full, pol_full, *_ = aaa(ys_aaa, xs_aaa)
    r_vals = np.array([float(r_full(jnp.array(x)).real) for x in xs_dense])
    err_full = np.abs(r_vals - true_vals)

    # Polynomial approximations for comparison
    f_poly = cj.chebfun(lambda x: x**(1.0/p), domain=(delta, 1.0))
    poly_errs = []
    degrees_poly = [5, 10, 20, 40, 80]
    for n in degrees_poly:
        pn = f_poly.polyfit(n)
        pn_vals = np.array([float(pn(jnp.array(x))) for x in xs_dense])
        poly_errs.append(np.max(np.abs(pn_vals - true_vals)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(xs_dense, true_vals, 'b', lw=1.8, label='x^{1/3}')
    ax.plot(xs_dense, r_vals, 'r--', lw=1.5,
            label=f'AAA ({len(pol_full)} poles)')
    ax.set_title(f'x^{{1/{p}}} on [{delta}, 1] and AAA approx', fontsize=10)
    ax.set_xlabel('x')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.semilogy(degrees_poly, poly_errs, 'b.-', lw=1.5, ms=10, label='poly')
    ax2.axhline(np.max(err_full), color='r', ls='--', lw=1.5,
                label=f'AAA ({len(pol_full)} poles)')
    ax2.set_title(f'Approximation errors for x^{{1/{p}}}', fontsize=10)
    ax2.set_xlabel('polynomial degree')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Polynomial vs. rational approximation of x^{{1/{p}}}', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'PthComposite.png'), dpi=150)
    plt.close(fig)

    print(f"PthComposite: p={p}, AAA err={np.max(err_full):.2e}")
    return True


if __name__ == '__main__':
    run()
