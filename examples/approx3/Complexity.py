"""Complexity of constructing 1D, 2D and 3D chebfuns.

Translation of approx3/Complexity.m by Nick Trefethen (April 2015):
timings of the chebfun, chebfun2, chebfun3 and chebfun3t constructors
as functions of the length m of the resulting approximation, for the
hard tanh ridge functions and the low-rank-friendly Runge functions.

Original: https://www.chebfun.org/examples/approx3/Complexity.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebfun3d.chebfun3t import chebfun3t
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx3')

MS = 7


def _kk(a, b):
    """2.^(a:.3333:b)."""
    return 2.0 ** np.arange(a, b + 1e-12, 0.3333)


def _timing_axes():
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    ax.grid(True, which="major", linestyle="-", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)
    ax.set_xlabel("length $m$")
    ax.set_ylabel("time")
    return fig, ax


def _save(fig, name):
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, name))
    plt.close(fig)


def _time3(ctor, g, kk):
    tt, mm = [], []
    for k in kk:
        t0 = time.time()
        f = ctor(lambda x, y, z, k=k: g(k, x, y, z))
        tt.append(time.time() - t0)
        mm.append(_len3(f)[0])
    return np.asarray(mm), np.asarray(tt)


def _len3(f):
    """[m n p] = length(f) for a chebfun3 or chebfun3t."""
    if hasattr(f, "length"):
        return f.length()
    return tuple(int(v) for v in f.coeffs.shape)


def _tanh3(k, x, y, z):
    return jnp.tanh(k * (x + y + z) / jnp.sqrt(3.0))


def _runge3(k, x, y, z):
    return 1.0 / (1 + k * (x**2 + y**2 + z**2))


def run():
    os.makedirs(_IMG, exist_ok=True)

    # 1D: tanh(k x), maxLength 2e6.
    tt, mm = [], []
    for k in _kk(3, 13.7):
        t0 = time.time()
        f = cj.chebfun(lambda x, k=k: jnp.tanh(k * x), max_length=2000000)
        tt.append(time.time() - t0)
        mm.append(len(f))
    mm, tt = np.asarray(mm), np.asarray(tt)
    fig, ax = _timing_axes()
    ax.loglog(mm, tt, '.b', ms=MS)
    ax.loglog(mm, 5e-6 * mm, '--b', lw=1)
    ax.axis([1e2, 1e6, 3e-3, 1])
    ax.text(1.1e4, .2, '$O(m)$', fontsize=11, color='b')
    ax.set_title('1D: $t = O(m)$')
    _save(fig, "Complexity_01.png")

    # 2D: tanh(k (x+y)/sqrt(2)).
    tt, mm = [], []
    for k in _kk(0, 5.4):
        t0 = time.time()
        f = cj.chebfun2(
            lambda x, y, k=k: jnp.tanh(k * (x + y) / jnp.sqrt(2.0)))
        tt.append(time.time() - t0)
        mm.append(f.length()[0])
    mm, tt = np.asarray(mm), np.asarray(tt)
    fig, ax = _timing_axes()
    ax.loglog(mm, tt, '.b', ms=MS)
    ax.loglog(mm, 1e-7 * mm.astype(float)**3, '--b', lw=1)
    ax.axis([1e1, 1e3, 2e-2, 1e2])
    ax.text(1.1e2, 1.1, '$O(m^3)$', fontsize=11, color='b')
    ax.set_title('2D: $t = O(m^3)$ ?')
    _save(fig, "Complexity_02.png")

    # 3D: tanh(k (x+y+z)/sqrt(3)) with chebfun3 ...
    mm3, tt3 = _time3(chebfun3, _tanh3, _kk(0, 3.4))
    fig, ax = _timing_axes()
    ax.loglog(mm3, tt3, '.b', ms=MS)
    ax.loglog(mm3, 2.5e-7 * mm3.astype(float)**4, '--b', lw=1)
    ax.axis([2e1, 2e2, 2e-2, 1e2])
    ax.text(30, 2, '$O(m^4)$', fontsize=11, color='b')
    ax.set_title('3D: $t = O(m^4)$ ?')
    ax.set_xticks([25, 50, 100], labels=["25", "50", "100"])
    _save(fig, "Complexity_03.png")

    # ... and with chebfun3t, overlaid on the same axes.
    mmt, ttt = _time3(chebfun3t, _tanh3, _kk(0, 3.4))
    fig, ax = _timing_axes()
    ax.loglog(mm3, tt3, '.b', ms=MS)
    ax.loglog(mm3, 2.5e-7 * mm3.astype(float)**4, '--b', lw=1)
    ax.axis([2e1, 2e2, 2e-2, 1e2])
    ax.text(30, 2, '$O(m^4)$', fontsize=11, color='b')
    ax.set_title('3D: $t = O(m^4)$ ?')
    ax.set_xticks([25, 50, 100], labels=["25", "50", "100"])
    ax.loglog(mmt, ttt, '.r', ms=MS)
    ax.loglog(mmt, 4e-6 * mmt.astype(float)**3, '--r', lw=1)
    ax.text(1.2e2, 4.4, '$O(m^3)$', fontsize=11, color='r')
    _save(fig, "Complexity_04.png")

    # Runge functions 1/(1 + k (x^2+y^2+z^2)): chebfun3 and chebfun3t.
    mm3, tt3 = _time3(chebfun3, _runge3, _kk(0, 6))
    mmt, ttt = _time3(chebfun3t, _runge3, _kk(0, 3.4))
    fig, ax = _timing_axes()
    ax.loglog(mm3, tt3, '.b', ms=MS)
    ax.loglog(mm3, 3e-5 * mm3.astype(float)**2, '--b', lw=1)
    ax.axis([35, 400, 0.1, 10])
    ax.text(115, .3, '$O(m^2)$', fontsize=11, color='b')
    ax.set_title('reversed behavior for Runge functions')
    ax.loglog(mmt, ttt, '.r', ms=MS)
    ax.loglog(mmt, 3e-6 * mmt.astype(float)**3, '--r', lw=1)
    ax.text(1.2e2, 4.4, '$O(m^3)$', fontsize=11, color='r')
    ax.set_xticks([50, 100, 200], labels=["50", "100", "200"])
    _save(fig, "Complexity_05.png")


if __name__ == "__main__":
    run()
