"""Composite rational approximation of pth roots.

Faithful replica of approx/PthComposite.m by Evan Gawlik and Yuji
Nakatsukasa (May 2019): Newton-type composite rational approximants of
x^(1/p) compared with minimax approximants.

Original: https://www.chebfun.org/examples/approx/PthComposite.html
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

P = 3
F = lambda x: x ** (1.0 / P)  # noqa: E731


def composite(alp0, k=3):
    alp = alp0
    rr = lambda x: np.ones_like(np.asarray(x, dtype=np.float64))  # noqa: E731
    for _ in range(k):
        mu = ((alp - alp**P) / ((P - 1) * (1 - alp))) ** (1.0 / P)
        rr = (lambda rprev, m: lambda x: (1.0 / P) * (
            (P - 1) * m * rprev(x)
            + np.asarray(x) / (m * rprev(x)) ** (P - 1)))(rr, mu)
        alp = P * alp / ((P - 1) * mu + mu ** (1 - P) * alp**P)
    return (lambda x: 2 * alp / (1 + alp) * rr(x)), alp


def _semilog(xx, yy, title, fname, vline=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.semilogx(xx, yy, lw=1.2)
    ax.grid(True)
    if vline is not None:
        ax.axvline(vline, ls='--', color='r', lw=1)
        ax.text(vline * 1.3, -vmax, r"$\alpha^p$", color='r', fontsize=15)
    ax.set_title(title, fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r55 = minimax(F, 5, rational=True, denom=5, domain=(0.0, 1.0))
    xx = np.logspace(-15, 0, 1000)
    _semilog(xx, F(xx) - np.asarray(r55.r(xx)),
             "absolute error of type (5,5) minimax approximant",
             "PthComposite_repl_01.png")

    # Composite approximant, k = 3, alpha = 0.03
    y, _ = composite(0.03)
    xr = np.logspace(np.log10(0.03**P), 0, 1000)
    _semilog(xr, (y(xr) - F(xr)) / F(xr),
             r"relative error of composite approximant, k=3, $\alpha$=0.03",
             "PthComposite_repl_02.png")
    abserr = y(xx) - F(xx)
    vmax = np.max(np.abs(abserr))
    _semilog(xx, abserr,
             r"absolute error of composite approximant, k=3, $\alpha$=0.03",
             "PthComposite_repl_03.png", vline=0.03**P, vmax=vmax)

    for i, alp0 in enumerate((0.1, 0.01)):
        y_, _ = composite(alp0)
        _semilog(xx, y_(xx) - F(xx),
                 rf"absolute error of composite approximant, k=3, "
                 rf"$\alpha$={alp0:g}",
                 f"PthComposite_repl_{i+4:02d}.png",
                 vline=alp0**P, vmax=vmax)

    # Composite (alpha = 0.03) vs minimax type (9,8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r98 = minimax(F, 9, rational=True, denom=8, domain=(0.0, 1.0))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.semilogx(xx, abserr, 'b', lw=1.2)
    ax.semilogx(xx, np.asarray(r98.r(xx)) - F(xx), 'r', lw=1.2)
    ax.grid(True)
    ax.text(xx[19], abserr[0] + 4e-3, "composite", color='b')
    ax.text(xx[19], (np.asarray(r98.r(xx)) - F(xx))[0] + 4e-3,
            "minimax", color='r')
    ax.set_title("composite vs. minimax approximants type (9,8), "
                 "absolute error", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "PthComposite_repl_06.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"minimax (5,5) err: {r55.err:.6e}")
    print(f"minimax (9,8) err: {r98.err:.6e}")

    # Cost comparison: Stahl's estimate vs composite scaling
    k = 10
    n = np.arange(1, 31)
    stahl = np.exp(-2 * np.pi * np.sqrt(n)) * 4 ** (1 + 1.0 / P) \
        * np.sin(np.pi / P)
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.semilogy(2 * n, stahl, '.-', ms=10)
    nn = float(P) ** np.arange(0, k + 1)
    exponent = (np.log(P / (P - 1)) * np.log(2)
                / (np.log(2 * P / (P - 1)) * np.log(P)))
    b = 3.0
    ax.semilogy(P * np.arange(1, k + 2), np.exp(-b * nn**exponent) * 10,
                '.-', ms=10)
    ax.grid(True)
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("error")
    ax.text(P * (k / 2), np.exp(-b * nn[0] ** (1 / exponent)),
            "composite", fontsize=11)
    ax.text(40, stahl[-1] * 10, "minimax", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "PthComposite_repl_07.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
