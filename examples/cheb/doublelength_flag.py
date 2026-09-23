"""The doublelength flag.

Translation of cheb/DoublelengthFlag.m by Nick Trefethen (July
2019): constructing chebfuns at twice the normally selected length to
inspect the rounding plateau.

Original: https://www.chebfun.org/examples/cheb/DoublelengthFlag.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')


def _coeffs(g, trig):
    """(index, |coefficient|) as MATLAB plotcoeffs shows them."""
    c = np.abs(np.asarray(g.coeffs))
    if trig:
        n = len(c)                        # wave numbers -(n-1)/2 .. (n-1)/2
        return np.arange(n) - (n - 1) // 2, c
    return np.arange(len(c)), c


def _pair_plot(f, f2, fname, fmt2, fmt1, trig=False):
    """plotcoeffs(f2,fmt2), hold on, plotcoeffs(f,fmt1), hold off."""
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    for g, fmt in ((f2, fmt2), (f, fmt1)):
        k, c = _coeffs(g, trig)
        kw = dict(color='r') if 'r' in fmt else {}
        if fmt.strip('r') == 'o':
            kw.update(marker='o', ms=10, mfc='none', mew=1.5, ls='none')
        elif fmt.strip('r') == '.':
            kw.update(marker='.', ms=(6 if trig else 7), ls='none')
        else:
            kw.update(lw=1.6)
        if not fmt.strip('r') and g is f2:
            kw.update(marker='.', ms=3, ls='none')
        ax.semilogy(k, c, **kw)
    ax.set_ylim(1e-20, 1)
    ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 1])
    if trig:
        n = (len(f2.coeffs) - 1) // 2
        ax.set_xlim(-n, n)
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', ls='--', alpha=0.2)
    ax.set_xlabel("Wave number" if trig else "Degree of Chebyshev polynomial")
    ax.set_ylabel("Magnitude of coefficient")
    ax.set_title("Fourier coefficients" if trig else "Chebyshev coefficients")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, fname))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # 'doublelength' (MATLAB @chebfun/chebfun.m): construct adaptively, then
    # again at the fixed length 2*length(f)-1.
    f = cj.chebfun('exp(x)')
    f2 = cj.chebfun('exp(x)', n=2 * len(f) - 1)
    _pair_plot(f, f2, "DoublelengthFlag_01.png", '.', 'or')

    f = cj.chebfun('sin(x)+sin(x^2)', domain=[0, 10])
    f2 = cj.chebfun('sin(x)+sin(x^2)', domain=[0, 10], n=2 * len(f) - 1)
    _pair_plot(f, f2, "DoublelengthFlag_02.png", '', 'r')

    ff = lambda t: 1.0 / (2 - jnp.cos(17 * (t - 1)))  # noqa: E731
    f = cj.chebfun(ff, domain=[-np.pi, np.pi], trig=True)
    f2 = cj.chebfun(ff, domain=[-np.pi, np.pi], trig=True, n=2 * len(f) - 1)
    _pair_plot(f, f2, "DoublelengthFlag_03.png", '.', '.r', trig=True)


if __name__ == "__main__":
    run()
