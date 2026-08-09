"""A Taylor's theorem analogue for Chebyshev series.

Faithful replica of temp/TaylorsTheorem.m (Hrothgar and Anthony
Austin, 2015): Chebyshev series converge in Bernstein ellipses --
truncated Chebyshev approximants of sin on growing grids converge
everywhere (entire function); for log|z-i| the extrapolation outside
the interval of approximation is limited by the largest Bernstein
ellipse avoiding the singularity, visualized via the Joukowski map.

Original: https://www.chebfun.org/examples/temp/TaylorsTheorem.html
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

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'temp')
FIG = [0]

RED = (.8, .3, .2)
BLUE = (.2, .3, .8)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"TaylorsTheorem_repl_{FIG[0]:02d}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)


def _trunc_cheb(fn, a, b, n):
    """Chebyshev series of fn on [a, b] truncated to n terms
    (MATLAB chebfun(..., 'trunc', n))."""
    f = chebfun(fn, domain=(a, b))
    tech = f.funs[0].tech
    c = np.zeros(n)
    cf = np.asarray(tech.coeffs).real
    m = min(n, cf.shape[0])
    c[:m] = cf[:m]

    def ev(x):
        t = (2 * np.asarray(x, dtype=float) - (a + b)) / (b - a)
        return np.polynomial.chebyshev.chebval(t, c)

    return ev


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # --- entire function: sin ---
    x0 = 0
    xs = np.linspace(-7, 7, 1400)
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    for k in range(2, 9):
        p = _trunc_cheb(np.sin, x0 - np.pi / 2, x0 + np.pi / 2, 2 * k + 1)
        ax.plot(xs, p(xs), '-',
                color=(.5 - k / 20, 1 - k / 10, .5 - k / 20))
    xi = np.linspace(x0 - np.pi / 2, x0 + np.pi / 2, 300)
    ax.plot(xi, p(xi), '-', color=(0, .6, 0), lw=5)
    ax.plot([x0], [np.sin(x0)], 'k.')
    ax.plot(xs, np.sin(xs), 'k-')
    ax.set_aspect("equal")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-3, 3)
    _save(fig)

    # --- non-entire analytic function: log|z - i| ---
    def func(x):
        return np.log(np.abs(np.asarray(x, dtype=float) - 1j))

    xs = np.linspace(-5, 5, 1400)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, func(xs), 'k-')
    _save(fig)

    x0, r1 = 2.0, 0.5
    p1 = _trunc_cheb(func, x0 - r1, x0 + r1, 60)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, func(xs), 'k-')
    ax.plot(xs, p1(xs), color=RED)
    xi = np.linspace(x0 - r1, x0 + r1, 200)
    ax.plot(xi, p1(xi), '-', color=RED, lw=5)
    ax.plot([x0], [func(x0)], 'k.')
    ax.set_ylim(-0.5, 2)
    _save(fig)

    r2 = 1.2
    p2 = _trunc_cheb(func, x0 - r2, x0 + r2, 80)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, func(xs), 'k-')
    ax.plot(xs, p2(xs), '-', color=BLUE)
    xi2 = np.linspace(x0 - r2, x0 + r2, 260)
    ax.plot(xi2, p2(xi2), '-', color=BLUE, lw=4)
    ax.plot(xi, p1(xi), '-', color=RED, lw=5)
    ax.plot([x0], [func(x0)], 'k.')
    ax.set_ylim(-0.5, 2)
    _save(fig)

    # Radii of convergence from the inverse Joukowski map.
    def invJ(z):
        return z + np.sqrt(z**2 - 1)

    def J(z):
        return (z + 1 / z) / 2

    sing = 1j
    rho1 = abs(invJ((sing - x0) / r1))
    d1 = J(rho1) * r1
    rho2 = abs(invJ((sing - x0) / r2))
    d2 = J(rho2) * r2
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(xs, func(xs), 'k-')
    ax.plot(xs, p2(xs), '-', color=BLUE)
    ax.plot(xi2, p2(xi2), '-', color=BLUE, lw=4)
    ax.plot(xi, p1(xi), '-', color=RED, lw=5)
    for d, c in [(d1, 'r--'), (d2, 'b--')]:
        ax.plot([x0 - d.real, x0 - d.real], [-0.5, 2], c)
        ax.plot([x0 + d.real, x0 + d.real], [-0.5, 2], c)
    ax.set_ylim(-0.5, 2)
    _save(fig)
    print(f"rho1 = {rho1:.6f}, radius of convergence d1 = {d1.real:.6f}")
    print(f"rho2 = {rho2:.6f}, radius of convergence d2 = {d2.real:.6f}")

    # --- Bernstein ellipses in the complex plane ---
    th = np.linspace(-1, 1, 800)
    z = np.exp(1j * np.pi * th)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    x0, r = 2.0, 1.6
    rho = abs(invJ((sing - x0) / r))
    e = J(rho * z)
    col = tuple(np.clip(1.25 * (1.5 - r / 2) * np.array(BLUE), 0, 1))
    ax.plot(e.real, e.imag, '-', color=col)
    s = (sing - x0) / r
    ax.plot([s.real], [s.imag], 'ko', markerfacecolor='none')
    ax.plot([-1, 1], [0, 0], 'k-')
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 2)
    _save(fig)

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for r in np.arange(1.0, 3.01, 0.3):
        rho = abs(invJ((sing - x0) / r))
        e = J(rho * z)
        col = tuple(np.clip(1.25 * (1.5 - r / 2) * np.array(BLUE), 0, 1))
        ax.plot(e.real, e.imag, '-', color=col)
        s = (sing - x0) / r
        ax.plot([s.real], [s.imag], 'ko', markerfacecolor='none')
    ax.plot([-1, 1], [0, 0], 'k-')
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 2)
    _save(fig)


if __name__ == "__main__":
    run()
