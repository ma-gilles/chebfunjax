"""Restricted-denominator approximations.

Explores rational approximation when the denominator is restricted to
a specific form, with applications to stability-preserving approximation
of the matrix exponential.

Credit: Stefan Guettel, April 2012.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/RestrictedDenominatorApproximations.html
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

    # Approximate exp(x) on [-1, 1] using restricted denominator
    # Denominator q(x) = (1 - x/2)^2 (simplest A-stable form)
    def restricted_approx(n_p):
        """Fit numerator p of degree n_p with fixed denominator q=(1-x/2)^2."""
        xs = np.cos(np.pi * np.arange(n_p + 3) / (n_p + 2))[::-1]
        ys = np.exp(xs)
        q_xs = (1 - xs / 2.0)**2
        # Fit p(x) = q(x) * exp(x) using polyfit
        p_coeffs = np.polyfit(xs, ys * q_xs, n_p)
        def r(x):
            return np.polyval(p_coeffs, x) / (1 - x / 2.0)**2
        return r

    xs_plot = np.linspace(-1.0, 1.0, 400)
    true_exp = np.exp(xs_plot)

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.plot(xs_plot, true_exp, 'k', lw=1.8, label='exp(x)')
    colors = ['b', 'r', 'g']
    for n, col in zip([3, 5, 8], colors):
        r = restricted_approx(n)
        r_vals = r(xs_plot)
        ax.plot(xs_plot, r_vals, '--', color=col, lw=1.2, label=f'p={n}')
    ax.set_title('Restricted-denominator approximants to exp(x)', fontsize=10)
    ax.legend(fontsize=8)
    ax2 = axes[1]
    for n, col in zip([3, 5, 8], colors):
        r = restricted_approx(n)
        err = np.abs(r(xs_plot) - true_exp)
        ax2.semilogy(xs_plot, err + 1e-18, '-', color=col, lw=1.3, label=f'p={n}')
    ax2.set_title('Error curves', fontsize=10)
    ax2.legend(fontsize=8)
    # Compare with AAA
    xs_aaa = jnp.linspace(-1.0, 1.0, 300)
    r_aaa, pol, *_ = aaa(jnp.exp(xs_aaa), xs_aaa)
    r_aaa_vals = np.array([float(r_aaa(jnp.array(x)).real) for x in xs_plot])
    ax2.semilogy(xs_plot, np.abs(r_aaa_vals - true_exp) + 1e-18, 'm-.',
                 lw=1.3, label=f'AAA')
    ax2.legend(fontsize=8)

    for ax in axes:
        pass
    fig.suptitle('Restricted-denominator rational approximations', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'RestrictedDenominatorApproximations.png'), dpi=150)
    plt.close(fig)

    print("RestrictedDenominatorApproximations: done.")
    return True

if __name__ == '__main__':
    run()