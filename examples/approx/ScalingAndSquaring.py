"""Rational approximation to the exponential in a complex region.

The scaling-and-squaring method uses the identity exp(A) = exp(A/2^s)^{2^s}
together with rational approximation to compute matrix exponentials efficiently.

Credit: Yuji Nakatsukasa and Stefan Guettel, July 2012.
Original MATLAB Chebfun: https://www.chebfun.org/examples/approx/ScalingAndSquaring.html
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


def pade_coeffs(m):
    """Compute [m/m] Padé numerator and denominator coefficients for exp(x).

    Returns p_coeffs, q_coeffs as arrays for np.polyval (highest degree first).
    Uses the standard formula: p_k = C(m,k) * C(m+k,k) / (2m)! / (2m+1-k)
    Specifically: numerator coeff c_k = m!*(2m-k)!/(2m)!/k!/(m-k)!, sign alternates in denom.
    """
    import math
    # Padé [m/m] coefficients for exp(x)
    # p(x) = sum_{k=0}^{m} c_k x^k  where c_k = m!(2m-k)! / ((2m)! k! (m-k)!)
    # q(x) = p(-x)
    c = np.zeros(m + 1)
    binom_mm = math.comb(2 * m, m)
    fac_2m = math.factorial(2 * m)
    for k in range(m + 1):
        c[k] = math.comb(m, k) * math.factorial(2 * m - k) / fac_2m
    # p_coeffs in polyval order (highest power first): c[m], c[m-1], ..., c[0]
    p_coeffs = c[::-1]
    # q(x) = p(-x): odd-degree coefficients negate
    q_coeffs = p_coeffs.copy()
    for i, coef in enumerate(q_coeffs):
        # degree = m - i
        deg = m - i
        if deg % 2 == 1:
            q_coeffs[i] = -coef
    return p_coeffs, q_coeffs


def run():
    os.makedirs(_OUTDIR, exist_ok=True)

    # Padé approximants to exp(x): [m/m] order
    def make_pade(m):
        p, q = pade_coeffs(m)
        def r(x): return np.polyval(p, x) / np.polyval(q, x)
        return r

    xs_real = np.linspace(-6.0, 6.0, 500)
    true_exp = np.exp(xs_real)

    pade2 = make_pade(2)
    pade4 = make_pade(4)
    err22 = np.abs(pade2(xs_real) - true_exp)
    err44 = np.abs(pade4(xs_real) - true_exp)

    fig, axes = plt.subplots(1, 2)

    ax = axes[0]
    ax.semilogy(xs_real, err22 + 1e-18, 'b', lw=1.5, label='[2/2] Padé')
    ax.semilogy(xs_real, err44 + 1e-18, 'r', lw=1.5, label='[4/4] Padé')
    ax.set_title('Padé approximation errors for exp(x) on [-6, 6]', fontsize=10)
    ax.legend(fontsize=9)
    # Scaling and squaring: scale x by 1/2^s, apply Padé, then square
    s = 3  # scale by 2^3 = 8
    def scaled_pade2(x, s=3):
        y = pade2(x / 2**s)
        for _ in range(s):
            y = y * y
        return y

    err_ss = np.abs(scaled_pade2(xs_real) - true_exp)

    ax2 = axes[1]
    ax2.semilogy(xs_real, err22 + 1e-18, 'b', lw=1.5, label='[2/2] Padé')
    ax2.semilogy(xs_real, err_ss + 1e-18, 'r', lw=1.5,
                 label=f'[2/2] Padé + scaling s={s}')
    ax2.set_title('Scaling and squaring improvement', fontsize=10)
    ax2.legend(fontsize=9)
    fig.suptitle('Scaling-and-squaring for matrix exponential approximation',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_OUTDIR, 'ScalingAndSquaring.png'), dpi=150)
    plt.close(fig)

    print(f"ScalingAndSquaring: [4/4] Padé max err on [-6,6] = {np.max(err44):.2e}")
    return True


if __name__ == '__main__':
    run()
