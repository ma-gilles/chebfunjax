"""Conformal map to a square.

Faithful replica of complex/ConformalSquare.m by Toby Driscoll
(January 2013): the Schwarz-Christoffel map of the unit disk to a
square, built up by integrating f'(z) along rays and circles.

The main ray map w = cumsum(fprime(z)) uses chebfun cumsum with
splitting (like MATLAB, whose published output shows an unresolved
65537-point final piece; ours resolves the corner value to 2e-8).
The four rays that terminate at prevertices and the boundary circle
have inverse-square-root singularities (MATLAB handles them with
SINGFUN, printing accuracy warnings); here they are integrated with a
smoothstep substitution that regularizes the endpoints exactly.

Original: https://www.chebfun.org/examples/complex/ConformalSquare.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

ZPRE = np.array([1, 1j, -1, -1j])
FIG = [0]


def fprime(z):
    return ((1 - z / ZPRE[0]) * (1 - z / ZPRE[1])
            * (1 - z / ZPRE[2]) * (1 - z / ZPRE[3])) ** (-0.5)


def _cumquad(h, a, b, n=4001):
    """Cumulative integral of h over [a, b] with a smoothstep
    substitution u = a + (b-a)(3x^2 - 2x^3), which regularizes
    inverse-square-root endpoint singularities exactly."""
    x = np.linspace(0.0, 1.0, n)
    u = a + (b - a) * (3 * x**2 - 2 * x**3)
    du = (b - a) * 6 * x * (1 - x)
    with np.errstate(all="ignore"):
        v = h(u) * du
    v = np.where(np.isfinite(v), v, 0.0)
    dx = x[1] - x[0]
    cum = np.concatenate([[0.0 + 0j],
                          np.cumsum((v[1:] + v[:-1]) / 2) * dx])
    return u, cum


def _snapshot(fig):
    FIG[0] += 1
    fig.savefig(os.path.join(
        _IMG, f"ConformalSquare_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")


def _plot_cf(ax, cf, n=400, **kw):
    bps = list(cf.domain.breakpoints)
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, max(8, int(n * (b - a)
                                         / (bps[-1] - bps[0]))))
        v = np.asarray(cf(t))
        ax.plot(v.real, v.imag, **kw)


def run():
    os.makedirs(_IMG, exist_ok=True)

    z = cj.chebfun(lambda x: x.astype(complex), domain=(0.0, 1.0))
    g = cj.chebfun(lambda x: fprime(x.astype(complex)),
                   domain=(0.0, 1.0), splitting=True)
    w = g.cumsum()
    print("w =")
    print(repr(w))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 5.2))
    th = np.linspace(0, 2 * np.pi, 400)
    axL.plot(np.cos(th), np.sin(th), 'k', lw=2)
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

    # rays from the origin; singular directions via substitution
    for t in np.linspace(0, 2 * np.pi, 33):
        e = np.exp(1j * t)
        axL.plot(zr * np.cos(t), zr * np.sin(t), 'b', lw=1.5)
        if np.min(np.abs(e - ZPRE)) < 1e-9:
            _, cum = _cumquad(lambda s: fprime(s * e) * e, 0.0, 1.0)
            axR.plot(cum.real, cum.imag, 'b', lw=1.5)
        else:
            gt = cj.chebfun(
                lambda r, _e=e: fprime(r.astype(complex) * _e) * _e,
                domain=(0.0, 1.0))
            _plot_cf(axR, gt.cumsum(), color='b', lw=1.5)
    _snapshot(fig)

    w1 = complex(np.asarray(w(1.0)))
    corners = w1 * ZPRE[[0, 1, 2, 3, 0]]
    axR.plot(corners.real, corners.imag, 'k--', lw=2)
    axR.plot(corners.real[:4], corners.imag[:4], 'r.', ms=14)
    _snapshot(fig)

    # images of circles of different radii (all smooth)
    for r in [0.5, 0.6, 0.7, 0.8, 0.9, 0.97]:
        axL.plot(r * np.cos(th), r * np.sin(th), 'b', lw=1.5)
        gt = cj.chebfun(
            lambda t_, _r=r: fprime(_r * np.exp(1j * t_))
            * 1j * _r * np.exp(1j * t_),
            domain=(0.0, 2 * np.pi))
        F = gt.cumsum() + complex(np.asarray(w(r)))
        _plot_cf(axR, F, color='b', lw=1.5)
    _snapshot(fig)

    # the boundary itself: square-root singularities at the prevertices
    segs = []
    start = w1
    for k in range(4):
        a, b = k * np.pi / 2, (k + 1) * np.pi / 2
        _, cum = _cumquad(
            lambda p: fprime(np.exp(1j * p)) * 1j * np.exp(1j * p),
            a, b)
        segs.append(start + cum)
        start = segs[-1][-1]
    curve = np.concatenate(segs)
    axR.plot(curve.real, curve.imag, 'k', lw=2)
    _snapshot(fig)
    plt.close(fig)
    print(f"corner value w(1) = {w1.real:.8f} "
          f"(exact lemniscatic value 1.31102878)")


if __name__ == "__main__":
    run()
