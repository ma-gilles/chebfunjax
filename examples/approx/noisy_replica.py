"""Chebfuns of noisy functions.

Faithful replica of approx/Noisy.m by Nick Trefethen (July 2014,
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
    c = np.abs(np.asarray(f.coeffs)) + 1e-30
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.semilogy(np.arange(len(c)), c, 'ob', ms=ms, mfc='none')
    if ref is not None:
        cr = np.abs(np.asarray(ref.coeffs)) + 1e-30
        ax.semilogy(np.arange(len(cr)), cr, '.k', ms=10)
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.set_xlabel("degree of Chebyshev polynomial")
    ax.set_ylabel("magnitude of coefficient")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # Default construction fails to resolve the noise:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cj.chebfun(ff, max_length=2**16)
    if rec:
        print("Warning:", str(rec[-1].message)[:60])

    # eps 1e-6 succeeds:
    f = cj.chebfun(ff, eps=1e-6)
    print("f ="); print(repr(f))
    f2 = cj.chebfun(ff, n=2 * len(f) - 1)   # 'doublelength' reference
    _coeffplot(f, "Noisy_repl_01.png", ref=f2)

    f = cj.chebfun(ff, eps=1e-3)
    print("f ="); print(repr(f))
    _coeffplot(f, "Noisy_repl_02.png", ref=f2)

    f = cj.chebfun(ff, eps=1e-9)
    print("f ="); print(repr(f))
    _coeffplot(f, "Noisy_repl_03.png", ref=f2)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cj.chebfun(ff, eps=1e-12, max_length=2**16)
    if rec:
        print("Warning:", str(rec[-1].message)[:60])

    # Smooth deterministic 'noise': plain construction resolves it fully
    g = cj.chebfun(gg)
    _coeffplot(g, "Noisy_repl_04.png", ylim=(1e-18, 1e2), ms=4)
    for e, fn in ((1e-6, "Noisy_repl_05.png"), (1e-9, "Noisy_repl_06.png"),
                  (1e-12, "Noisy_repl_07.png")):
        gge = cj.chebfun(gg, eps=e)
        _coeffplot(gge, fn, ylim=(1e-18, 1e2), ms=4)
        print(f"eps={e:g}: len {len(gge)}")


if __name__ == "__main__":
    run()
