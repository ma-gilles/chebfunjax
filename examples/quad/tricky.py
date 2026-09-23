"""Some tricky integrals.

Translation of quad/Tricky.m by Fredrik Johansson and Nick Trefethen
(March 2017): integrals with spikes, high oscillation, jumps, a
boundary layer, an unbounded domain, kinks and a wildly oscillating
integrand, each computed as ``sum(chebfun(...))`` and compared with a
high-precision reference value.

Original: https://www.chebfun.org/examples/quad/Tricky.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jax.scipy.special import erf  # noqa: E402
from scipy.special import airy  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj  # noqa: E402
from chebfunjax.plotting import chebfun_style  # noqa: E402
from chebfunjax.plotting import save_chebfun_figure as _savefig  # noqa: E402

jax.config.update("jax_enable_x64", True)
chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')
FIG = [0]
_T0 = [0.0]


def tic():
    _T0[0] = time.time()


def toc():
    print(f"Elapsed time is {time.time() - _T0[0]:.6f} seconds.")


def show(name, v):
    """MATLAB display of ``name = v`` (format long)."""
    print(f"{name} =")
    if isinstance(v, cj.Chebfun):
        print(repr(v))
    elif v != v:
        print("   NaN")
    elif abs(v) >= 1000 and float(v).is_integer():
        print(f"        {int(v)}")
    else:
        pad = 3 - (v < 0) - (abs(v) >= 10)
        print(" " * pad + f"{v:.15f}")


def plot(f, window=None):
    """plot(f) of a (piecewise) chebfun on its domain (or ``window``)."""
    FIG[0] += 1
    a, b = window if window is not None else (float(f.domain.a), float(f.domain.b))
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    for p in f.funs:
        pa, pb = (max(float(p.interval[0]), a), min(float(p.interval[1]), b))
        if pb <= pa:
            continue
        m = max(64, min(4000, 4 * p.n))
        xs = np.linspace(pa, pb, m)
        ax.plot(xs, np.real(np.asarray(f(jnp.asarray(xs)))), color="C0", lw=1.0)
    ax.set_xlim(a, b)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Tricky_{FIG[0]:02d}.png"), size=(600, 270))
    plt.close(fig)


def _np_handle(fn):
    """Wrap a numpy-only special function as a vectorised handle."""
    def h(x):
        xa = np.asarray(x, dtype=float)
        return jnp.asarray(fn(xa))
    return h


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Three spikes.
    def ff(x):
        return (1 / jnp.cosh(10 * (x - .2)) ** 2 + 1 / jnp.cosh(100 * (x - .4)) ** 4
                + 1 / jnp.cosh(1000 * (x - .6)) ** 6)
    Iexact = 0.210802735500549277
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(ff, domain=[0, 1]); show("f", f); I = float(f.sum()); toc()
    plot(f)
    tic(); f = cj.chebfun(ff, domain=[0, 1], splitting=True); show("f", f); I = float(f.sum()); toc()

    # Oscillation.
    def ff(x):
        return jnp.sin(x + jnp.exp(x))
    Iexact = 0.34740017265724780787
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(ff, domain=[0, 8]); show("f", f); show("I", float(f.sum())); toc()
    plot(f)

    # Oscillation with jumps.
    def ff(x):
        return (jnp.exp(x) - jnp.floor(jnp.exp(x))) * jnp.sin(x + jnp.exp(x))
    Iexact = 0.098651704478365206119
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(ff, domain=[0, 8], splitting=True); show("I", float(f.sum())); toc()

    # A boundary layer.
    def ff(x):
        return jnp.exp(-x) * erf(jnp.sqrt(1250.0) * x + 1.5)
    Iexact = float("nan")
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(ff); show("f", f); show("I", float(f.sum())); toc()
    plot(f)

    # An unbounded domain.
    ff = _np_handle(lambda x: np.exp(-x) * airy(-x)[0])
    Iexact = 0.378751605379086535
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(ff, domain=[0, np.inf]); show("f", f); show("I", float(f.sum())); toc()
    plot(f, window=(0.0, 50.0))
    tic(); f = cj.chebfun(ff, domain=[0, 40]); show("f", f); show("I", float(f.sum())); toc()

    # A kinked integrand.
    def ff(x):
        return jnp.abs(x ** 4 + 10 * x ** 3 + 19 * x ** 2 - 6 * x - 6) * jnp.exp(x)
    Iexact = 11.1473105500571397339
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(ff, domain=[0, 1], splitting=True); show("f", f)
    show("I", float(f.sum())); toc()
    plot(f)

    # Jumps.
    Iexact = 5050.0
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(jnp.ceil, domain=[0, 100], splitting=True)
    show("I", float(f.sum())); toc()
    plot(f)

    # Jumps and kinks.
    def ff(x):
        return (x - jnp.floor(x) - 0.5) * jnp.maximum(jnp.sin(x), jnp.cos(x))
    Iexact = -0.14281864202632808376
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(ff, domain=[0, 10], splitting=True); show("I", float(f.sum())); toc()
    plot(f)

    # Wild oscillation near x = 1.
    def ff(x):
        return jnp.sin((0.001 + (1 - x) ** 2) ** (-1.5))
    Iexact = 0.74997436852719477011
    show("Iexact", Iexact)
    tic(); f = cj.chebfun(ff, domain=[0, 3], max_length=int(1e7)); show("f", f)
    show("I", float(f.sum())); toc()
    plot(f)
    tic(); f = cj.chebfun(ff, domain=[0, 3], splitting=True, split_max_length=int(1e6))
    show("I", float(f.sum())); toc()
    plot(f)


if __name__ == "__main__":
    run()
