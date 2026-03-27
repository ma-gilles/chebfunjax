"""Accuracy of Chebyshev coefficients via aliasing.

Illustrates how Chebyshev coefficients of a degree-n interpolant are related
to the exact coefficients via aliasing formulae.

Credit: Yuji Nakatsukasa, April 2016.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/AliasingCoefficients.html
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

    # Analytic function: log(sin(10x) + 2)
    def fori(x): return jnp.log(jnp.sin(10.0 * x) + 2.0)

    f = cj.chebfun(fori)
    n_low = max(5, len(f) // 3)
    p = cj.chebfun(fori, n=n_low)

    fc = np.array(f.coeffs)
    pc = np.array(p.coeffs)
    n_p = len(pc)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(np.arange(len(fc)), np.abs(fc) + 1e-18, '.g', ms=7,
                label='f coeffs')
    ax.semilogy(np.arange(n_p), np.abs(pc) + 1e-18, '.b', ms=7,
                label='p coeffs')
    err_coeffs = np.abs(pc - fc[:n_p]) + 1e-18
    ax.semilogy(np.arange(n_p), err_coeffs, '.r', ms=7,
                label='|f−p| coeffs (aliasing error)')
    ax.set_title('Aliasing of Chebyshev coefficients', fontsize=11)
    ax.set_xlabel('degree')
    ax.set_ylabel('|coefficient|')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'AliasingCoefficients.png'), dpi=150)
    plt.close(fig)

    print(f"AliasingCoefficients: len(f)={len(f)}, len(p)={len(p)}")
    return True


if __name__ == '__main__':
    run()
