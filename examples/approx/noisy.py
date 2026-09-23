"""Chebfuns of noisy functions.

Translation of approx/Noisy.m by Nick Trefethen (July 2014,
revised 2019): constructing chebfuns of functions contaminated by
1e-6-level noise using the 'eps' preference.

Original: https://www.chebfun.org/examples/approx/Noisy.html
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
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')


def ff(x):
    n = x.shape[0] if hasattr(x, "shape") and x.ndim else 1
    idx = jnp.arange(1, n + 1, dtype=jnp.float64)
    return jnp.tanh(8 * (x - 0.5)) + 1e-6 * jnp.sin(idx**2)


def gg(x):
    return jnp.tanh(8 * (x - 0.5)) + 1e-6 * jnp.sin(200 * jnp.exp(x))


def _coeffplot(f, fname, ref=None, ylim=(1e-10, 10), ms=7):
    # plotcoeffs(f, 'ob', MS, ms), ylim(...) [, hold on, plotcoeffs(f2, '.k')]
    c = np.abs(np.asarray(f.coeffs))
    fig, ax = plt.subplots(figsize=(5.98, 2.73))
    ax.semilogy(np.arange(len(c)), c, 'o', color='b', ms=ms / 1.5,
                mfc='none')
    n = len(c)
    if ref is not None:
        cr = np.abs(np.asarray(ref.coeffs))
        ax.semilogy(np.arange(len(cr)), cr, '.k', ms=ms / 1.5)
        n = max(n, len(cr))
    ax.set_xlim(0, n - 1)
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.set_title("Chebyshev coefficients")
    ax.set_xlabel("Degree of Chebyshev polynomial")
    ax.set_ylabel("Magnitude of coefficient")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, fname), size=(598, 273))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Default construction fails to resolve the noise:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cj.chebfun(ff, max_length=2**16 + 1)
    if rec:
        print("Warning:", str(rec[-1].message))

    # eps 1e-6 succeeds:
    f = cj.chebfun(ff, eps=1e-6)
    print("f ="); print(repr(f))
    f2 = cj.chebfun(ff, n=2 * len(f) - 1)   # 'doublelength' reference
    _coeffplot(f, "Noisy_01.png", ref=f2)

    f = cj.chebfun(ff, eps=1e-3)
    print("f ="); print(repr(f))
    _coeffplot(f, "Noisy_02.png", ref=f2)

    f = cj.chebfun(ff, eps=1e-9)
    print("f ="); print(repr(f))
    _coeffplot(f, "Noisy_03.png", ref=f2)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cj.chebfun(ff, eps=1e-12, max_length=2**16 + 1)
    if rec:
        print("Warning:", str(rec[-1].message))

    # Smooth deterministic 'noise': plain construction resolves it fully
    g = cj.chebfun(gg)
    _coeffplot(g, "Noisy_04.png", ylim=(1e-18, 1e2), ms=4)
    for e, fn in ((1e-6, "Noisy_05.png"), (1e-9, "Noisy_06.png"),
                  (1e-12, "Noisy_07.png")):
        gge = cj.chebfun(gg, eps=e)
        _coeffplot(gge, fn, ylim=(1e-18, 1e2), ms=4)


if __name__ == "__main__":
    run()
