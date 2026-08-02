"""The white curves of Ortiz and Rivlin.

Faithful replica of roots/WhiteCurves.m by Nick Trefethen
(November 2010): the white curves visible in plots of superimposed
Chebyshev polynomials T_1..T_30, given by T_{n-m}(x) = T_2(y), and
the analogous (envelope-corrected) picture for Legendre polynomials.

Original: https://www.chebfun.org/examples/roots/WhiteCurves.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.polynomials import legpoly

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'roots')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"WhiteCurves_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")


def _cheb_roots_real(coeffs):
    r = np.polynomial.chebyshev.chebroots(coeffs)
    r = r[np.abs(r.imag) < 1e-10].real
    return r[np.abs(r) <= 1 + 1e-10]


def run():
    os.makedirs(_IMG, exist_ok=True)
    xs = np.linspace(-1, 1, 1000)

    # T_1 .. T_30 superimposed
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for n in range(1, 31):
        ax.plot(xs, np.cos(n * np.arccos(xs)), 'b-', lw=0.7)
    ax.axis([-1, 1, -1, 1])
    _save(fig)

    # white curves: T_j(x) = T_2(y), j = 1..4
    T2 = lambda y: 2 * y**2 - 1  # noqa: E731
    for j in range(1, 5):
        cj_ = np.zeros(j + 1)
        cj_[j] = 1.0
        for y in np.linspace(-1, 1, 200):
            c = cj_.copy()
            c[0] -= T2(y)
            x = _cheb_roots_real(c)
            ax.plot(x, np.full_like(x, y), 'r.', ms=4)
    ax.axis([-1, 1, -1, 1])
    _save(fig)
    plt.close(fig)

    # Legendre analogue, with the (1-x^2)^{1/4} envelope correction
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    with np.errstate(invalid="ignore"):
        q = lambda n: np.sqrt(np.pi * n / 2) * (1 - xs**2)**0.25  # noqa: E731
    for n in range(1, 31):
        Lv = np.polynomial.chebyshev.chebval(
            xs, np.asarray(legpoly(n)))
        ax.plot(xs, Lv * q(n), color=(0.6, 0.4, 0), lw=0.7)
    ax.axis([-1, 1, -1, 1])
    _save(fig)
    plt.close(fig)

    # white curves for Legendre: L_j(x) = L_2(y)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    L2 = lambda y: (3 * y**2 - 1) / 2  # noqa: E731
    for j in range(1, 5):
        cj_ = np.array(np.asarray(legpoly(j)), dtype=float)
        for y in np.linspace(-1, 1, 200):
            cc = cj_.copy()
            cc[0] -= L2(y)
            x = _cheb_roots_real(cc)
            if len(x):
                ax.plot(x, np.full_like(x, y), 'r.', ms=4)
    ax.axis([-1, 1, -1, 1])
    _save(fig)
    plt.close(fig)
    print("figures:", FIG[0])


if __name__ == "__main__":
    run()
