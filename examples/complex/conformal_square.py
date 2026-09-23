"""Conformal map to a square.

Translation of complex/ConformalSquare.m by Toby Driscoll
(January 2013): the Schwarz-Christoffel map of the unit disk to a
square, built up by integrating f'(z) along rays and circles.

MATLAB's chebfun power of a complex chebfun (``columnPower``) first adds
breakpoints at the roots of its imaginary part and then raises each
piece to the power; ``fprime`` below does exactly that with chebfunjax
calls, because ``Chebfun.__pow__`` of a complex chebfun with a boundary
root raises (``Singfun.extractBoundaryRoots`` casts complex coefficients
to float).  The pieces ending at a prevertex are unresolved
65537-point pieces, as in the published output.

Original: https://www.chebfun.org/examples/complex/ConformalSquare.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

ZPRE = np.array([1, 1j, -1, -1j])
FIG = [0]


def fprime(z):
    P = ((1 - z / ZPRE[0]) * (1 - z / ZPRE[1])
         * (1 - z / ZPRE[2]) * (1 - z / ZPRE[3]))
    a, b = float(P.domain.breakpoints[0]), float(P.domain.breakpoints[-1])
    r = np.asarray(P.imag().roots()).ravel()
    r = r[(r > a + 1e-12) & (r < b - 1e-12)] if P.imag().norm() > 0 else []
    return cj.chebfun(lambda x: P(x) ** (-0.5),
                      domain=[a, *sorted(float(v) for v in r), b])


def _snapshot(fig):
    FIG[0] += 1
    _savefig(fig, os.path.join(
        _IMG, f"ConformalSquare_{FIG[0]:02d}.png"))


def _plot_cf(ax, cf, n=400, **kw):
    bps = list(cf.domain.breakpoints)
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, max(8, int(n * (b - a)
                                         / (bps[-1] - bps[0]))))
        v = np.asarray(cf(t))
        ax.plot(v.real, v.imag, **kw)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.simplefilter("ignore")

    z = cj.chebfun('z', domain=(0.0, 1.0))
    w = fprime(z).cumsum()
    print("w =")
    print(repr(w))

    zcirc = (1j * cj.chebfun('t', domain=(0.0, 2 * np.pi))).exp()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 5.2))
    _plot_cf(axL, zcirc, color='k', lw=2)
    axL.plot(ZPRE.real, ZPRE.imag, 'r.', ms=14)
    axL.set_aspect("equal")
    axL.axis(list(1.05 * np.array([-1, 1, -1, 1])))
    axL.set_axis_off()
    zr = np.linspace(0, 1, 100)
    axL.plot(zr, 0 * zr, 'b', lw=1.5)
    _plot_cf(axR, w, color='b', lw=1.5)
    axR.set_aspect("equal")
    axR.axis(list(1.35 * np.array([-1, 1, -1, 1])))
    axR.set_axis_off()
    fig.set_facecolor("white")
    _snapshot(fig)

    for t in np.linspace(0, 2 * np.pi, 33):
        e = np.exp(1j * t)
        zt = cj.chebfun('r', domain=(0.0, 1.0)) * e
        axL.plot(zr * np.cos(t), zr * np.sin(t), 'b', lw=1.5)
        _plot_cf(axR, (fprime(zt) * e).cumsum(), color='b', lw=1.5)
    _snapshot(fig)

    w1 = complex(np.asarray(w(1.0)))
    corners = w1 * ZPRE[[0, 1, 2, 3, 0]]
    axR.plot(corners.real, corners.imag, 'k--', lw=2)
    axR.plot(corners.real[:4], corners.imag[:4], 'r.', ms=14)
    _snapshot(fig)

    for r in [0.5, 0.6, 0.7, 0.8, 0.9, 0.97]:
        zr_ = r * zcirc
        f = (fprime(zr_) * zr_.diff()).cumsum() + complex(np.asarray(w(r)))
        _plot_cf(axL, zr_, color='b', lw=1.5)
        _plot_cf(axR, f, color='b', lw=1.5)
    _snapshot(fig)

    f = (fprime(zcirc) * zcirc.diff()).cumsum() + w1
    _plot_cf(axR, f, color='k', lw=2)
    _snapshot(fig)
    plt.close(fig)


if __name__ == "__main__":
    run()
