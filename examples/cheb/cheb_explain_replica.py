"""Explaining chebfun construction.

Faithful replica of cheb/ChebExplain.m by Nick Trefethen (March 2017):
annotated coefficient plots showing how the constructor selects grids
and chops series for a range of easy and awkward functions.

Original: https://www.chebfun.org/examples/cheb/ChebExplain.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')

FIG = [0]


def explain(fop, label, eps=None):
    """Simplified port of cheb.explain: the chopped chebfun's
    coefficients (blue) over the doublelength construction (grey),
    with the working tolerance marked."""
    FIG[0] += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = {} if eps is None else {"eps": eps}
        f = cj.chebfun(fop, **kw)
        f2 = cj.chebfun(fop, n=2 * len(f))
    c = np.abs(np.asarray(f.coeffs)) + 1e-30
    c2 = np.abs(np.asarray(f2.coeffs)) + 1e-30
    vscale = float(np.max(c))
    tol = (eps if eps is not None else 2.2e-16) * max(vscale, 1e-300)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.semilogy(np.arange(len(c2)), c2, '.', color="0.75", ms=6)
    ax.semilogy(np.arange(len(c)), c, '.b', ms=7)
    ax.axhline(tol, color='r', ls='--', lw=1)
    ax.text(0.02, 0.05, f"len = {len(f)}", transform=ax.transAxes)
    ax.grid(True)
    ax.set_title(f"explain('{label}')", fontsize=12)
    ax.set_xlabel("degree")
    ax.set_ylabel("|coefficient|")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"ChebExplain_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"explain('{label}'"
          + (f", {eps:g}" if eps is not None else "") + f"): len {len(f)}")


def run():
    os.makedirs(_IMG, exist_ok=True)

    explain(lambda x: 100000 * jnp.exp(x), "100000*exp(x)")
    explain(lambda x: jnp.exp(-(x - 0.5) ** 2), "exp(-(x-.5)^2)")
    explain(lambda x: 1.0 / (1 + 1000 * x**2), "1/(1+1000*x^2)")
    explain(lambda x: jnp.exp(x) + 1e-8 * jnp.cos(99 * x),
            "exp(x) + 1e-8*cos(99*x)")
    explain(lambda x: jnp.exp(x) + 1e-12 * jnp.cos(99 * x),
            "exp(x) + 1e-12*cos(99*x)")
    explain(lambda x: jnp.abs(x) ** 3, "abs(x)^3")

    # A degree-3000 interpolant of |x|^3
    f = cj.chebfun(lambda x: jnp.abs(x) ** 3, n=3000)
    print("f =")
    print(repr(f))

    # Unresolvable hidden oscillation at machine precision:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cj.chebfun(lambda x: jnp.exp(x) + 1e-8 * jnp.cos(99999 * x),
                   max_length=2**16)
    if rec:
        print("Warning:", str(rec[-1].message)[:60])

    # ... resolvable with eps = 1e-8:
    f = cj.chebfun(lambda x: jnp.exp(x) + 1e-8 * jnp.cos(99999 * x),
                   eps=1e-8)
    print("f =")
    print(repr(f))
    explain(lambda x: jnp.exp(x) + 1e-8 * jnp.cos(99999 * x),
            "exp(x) + 1e-8*cos(99999*x)", eps=1e-8)


if __name__ == "__main__":
    run()
