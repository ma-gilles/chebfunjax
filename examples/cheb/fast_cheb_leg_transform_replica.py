"""The fast Chebyshev-Legendre transform.

Replica of cheb/FastChebyshevLegendreTransform.m by Nick Hale and Alex
Townsend (August 2013): converting between Chebyshev and Legendre
expansions.  The coefficient-comparison sections replicate exactly;
the two large-N timing demos (N ~ 24000-32000) require the O(N log N)
transform of Hale & Townsend, which chebfunjax has not ported (its
cheb2leg/leg2cheb are quadratic-cost; ledgered), and are described
rather than run.

Original: https://www.chebfun.org/examples/cheb/FastChebyshevLegendreTransform.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.transforms import cheb2leg

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Runge-type function
    f = cj.chebfun(lambda x: 1.0 / (1 + 1000 * (x - 0.1) ** 2))
    c_cheb = np.asarray(f.coeffs)
    c_leg = np.asarray(cheb2leg(f.coeffs))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.abs(c_leg) + 1e-30, 'xr', ms=4,
                label="Legendre coefficients")
    ax.semilogy(np.abs(c_cheb) + 1e-30, '.b', ms=8,
                label="Chebyshev coefficients")
    ax.legend()
    ax.set_xlabel("n", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "FastChebyshevLegendreTransform_repl_01.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Runge-type: N = {len(f)}")

    # |x - .1|^(7/4): algebraic decay with a half-power gap between the
    # two coefficient families
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = cj.chebfun(lambda x: jnp.abs(x - 0.1) ** (7.0 / 4),
                       max_length=2**13)
    N = len(g)
    c_cheb = np.asarray(g.coeffs)
    c_leg = np.asarray(cheb2leg(g.coeffs))
    nn = np.arange(1, N + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(np.abs(c_leg) + 1e-30, 'xr', ms=4,
                label="Legendre coefficients")
    ax.semilogy(np.abs(c_cheb) + 1e-30, '.b', ms=8,
                label="Chebyshev coefficients")
    ax.semilogy(nn, nn ** (-7.0 / 4 - 1 + 0.5), 'k--', lw=1.6,
                label=r"$O(n^{-2.25})$")
    ax.semilogy(nn, nn ** (-7.0 / 4 - 1), 'k--', lw=1.6,
                label=r"$O(n^{-2.75})$")
    ax.set_xlim(0, N)
    ax.set_xlabel("n", fontsize=12)
    ax.legend()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "FastChebyshevLegendreTransform_repl_02.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"|x-.1|^(7/4): N = {N}")


if __name__ == "__main__":
    run()
