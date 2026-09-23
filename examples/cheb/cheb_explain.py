"""Explaining chebfun construction.

Translation of cheb/ChebExplain.m by Nick Trefethen (March 2017):
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
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'cheb')

FIG = [0]
EPS = 2.0 ** -52


def _standard_chop(coeffs, tol):
    """explain.m's local copy of standardChop, which also returns
    plateauPoint and j2 (1-based, as in MATLAB)."""
    n = len(coeffs)
    plateau_point, j2 = n, n
    if tol >= 1:
        return 1, plateau_point, j2
    cutoff = n
    if n < 17:
        return cutoff, plateau_point, j2
    # Step 1: monotonically nonincreasing envelope normalized to 1.
    m = np.maximum.accumulate(np.abs(coeffs)[::-1])[::-1]
    if m[0] == 0:
        return 1, plateau_point, j2
    envelope = m / m[0]
    # Step 2: find the plateau point.
    for j in range(2, n + 1):
        j2 = int(np.floor(1.25 * j + 5 + 0.5))
        if j2 > n:
            return cutoff, plateau_point, j2
        e1, e2 = envelope[j - 1], envelope[j2 - 1]
        r = 3 * (1 - np.log(e1) / np.log(tol))
        if e1 == 0 or e2 / e1 > r:
            plateau_point = j - 1
            break
    # Step 3: chop where the tilted envelope is minimal.
    if envelope[plateau_point - 1] == 0:
        return plateau_point, plateau_point, j2
    j3 = int(np.sum(envelope >= tol ** (7 / 6)))
    if j3 < j2:
        j2 = j3 + 1
        envelope[j2 - 1] = tol ** (7 / 6)
    cc = np.log10(envelope[:j2]) + np.linspace(0, (-1 / 3) * np.log10(tol),
                                               j2)
    d = int(np.argmin(cc)) + 1
    return max(d - 1, 1), plateau_point, j2


def _basic_chebfun(ff, tol):
    """explain.m's simplified constructor on grids of 17, 33, ..., 65537."""
    for ii in range(4, 17):
        m = 2 ** ii + 1
        cfs = np.asarray(Chebtech2.vals2coeffs(ff(chebpts(m))))
        cutoff, pp, j2 = _standard_chop(cfs, tol)
        if cutoff < m:
            break
    return cfs[:cutoff], cfs, pp, j2


def explain(ff, label, epsval=EPS):
    """cheb.explain: how standardChop chops the Chebyshev series of ff.

    Black dots: coefficients on the finest grid sampled; red circles:
    coefficients kept; green: the envelope; dashed: the tolerance;
    blue square: plateauPoint; magenta: the tilted ruler of Step 3."""
    FIG[0] += 1
    fc, gc, pp, j2 = _basic_chebfun(ff, epsval)
    fc, gc = np.abs(fc), np.abs(gc)
    nf, ng = len(fc), len(gc)
    ms, ms0 = 3, 5
    if ng < 1000:
        ms, ms0 = 5, 4.5
    if ng < 100:
        ms, ms0 = 7, 5
    m = np.maximum.accumulate(gc[::-1])[::-1]

    # Axis limits of a first semilogy of envelope and coefficients,
    # widened as in explain.m.
    pos = np.concatenate([m[m > 0], gc[gc > 0], fc[fc > 0]])
    lo = 10.0 ** np.floor(np.log10(pos.min()))
    hi = 10.0 ** np.ceil(np.log10(pos.max()))
    xmax = MaxNLocator(nbins=8).tick_values(0, ng - 1)[-1]
    a = [0.0, max(xmax, ng), 1e-3 * lo, 1e3 * hi]
    m = np.where(m == 0, a[2], m)

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    k = np.arange(ng)
    ax.semilogy(k, m, '-', color=(0, 1, 0))
    ax.semilogy(k, 0 * m + epsval * m[0], 'k--')
    ax.semilogy(k, gc, '.k', ms=ms)
    ax.semilogy(np.arange(nf), fc, 'o', color='r', mfc='none', ms=ms0)
    ax.axis(a)
    ax.tick_params(labelsize=9)
    xpos = a[0] + .95 * (a[1] - a[0])
    ypos = np.exp(np.log(a[2]) + .86 * (np.log(a[3]) - np.log(a[2])))
    ax.text(xpos, ypos, label, fontsize=13, ha='right')
    ax.plot(pp - 1, gc[pp - 1], 's', color='b', mfc='none', ms=10)
    tilt = (1 / 3) * np.log10(epsval) / (j2 - 1)
    ruler = 10.0 ** (tilt * np.arange(j2))
    ruler = ruler * m[min(nf, ng - 1)] / ruler[min(nf, j2 - 1)]
    ax.semilogy(np.arange(j2), ruler, 'm')
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"ChebExplain_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    explain(lambda x: 100000 * jnp.exp(x), r"$100000\,\exp(x)$")
    explain(lambda x: jnp.exp(-(x - 0.5) ** 2), r"$\exp(-(x-.5)^2)$")
    explain(lambda x: 1.0 / (1 + 1000 * x**2), r"$1/(1+1000\,x^2)$")
    explain(lambda x: jnp.exp(x) + 1e-8 * jnp.cos(99 * x),
            r"$\exp(x) + 10^{-8}\,\cos(99\,x)$")
    explain(lambda x: jnp.exp(x) + 1e-12 * jnp.cos(99 * x),
            r"$\exp(x) + 10^{-12}\,\cos(99\,x)$")
    explain(lambda x: jnp.abs(x) ** 3, r"$|x|^3$")

    f = cj.chebfun(lambda x: jnp.abs(x) ** 3, n=3000)
    print("f =")
    print(repr(f))

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f = cj.chebfun(lambda x: jnp.exp(x) + 1e-8 * jnp.cos(99999 * x))
    for w in rec:
        print("Warning:", str(w.message))
    print("f =")
    print(repr(f))

    f = cj.chebfun(lambda x: jnp.exp(x) + 1e-8 * jnp.cos(99999 * x),
                   eps=1e-8)
    print("f =")
    print(repr(f))
    explain(lambda x: jnp.exp(x) + 1e-8 * jnp.cos(99999 * x),
            r"$\exp(x) + 10^{-8}\,\cos(99999\,x)$", 1e-8)


if __name__ == "__main__":
    run()
