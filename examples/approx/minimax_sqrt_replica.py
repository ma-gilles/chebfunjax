"""Minimax approximation of sqrt(x).

Faithful replica of approx/MinimaxSqrt.m by Yuji Nakatsukasa and Nick
Trefethen (October 2019): polynomial vs rational minimax approximation
of sqrt on [a,1] as a shrinks to 0 — rational approximations barely
degrade while polynomials fall apart.

Original: https://www.chebfun.org/examples/approx/MinimaxSqrt.html
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.minimax import minimax

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

F = lambda x: jnp.sqrt(x)  # noqa: E731


def _errs(a, ns):
    perrs, rerrs = [], []
    for n in ns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                perrs.append(minimax(F, n, domain=(a, 1.0)).err)
            except Exception:
                perrs.append(np.nan)
            try:
                rerrs.append(minimax(F, n // 2, rational=True,
                                     denom=n // 2,
                                     domain=(a, 1.0)).err)
            except Exception:
                rerrs.append(np.nan)
    return np.array(perrs), np.array(rerrs)


def _figure(a, ns, perrs, rerrs, fname):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.semilogy(ns, perrs, 'b*-', lw=1.2)
    ax.text(ns[-1] + 0.4, perrs[-1], f"poly a={a:g}", color='b')
    ax.semilogy(ns, rerrs, 'r*-', lw=1.2)
    ax.text(ns[-1] + 0.4, rerrs[-1], f"rat a={a:g}", color='r')
    ax.grid(True)
    ax.set_xlim(0, ns[-1] + (2 if ns[-1] <= 8 else 7))
    ax.set_title(f"sqrt(x) on [a,1], a = {a:g}", fontsize=12)
    ax.set_xlabel("DOF")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    ns_small = np.arange(2, 9, 2)
    for i, a in enumerate((0.8, 0.1)):
        p, r = _errs(a, ns_small)
        _figure(a, ns_small, p, r, f"MinimaxSqrt_repl_{i+1:02d}.png")
        print(f"a={a}: poly {p[-1]:.3e}  rat {r[-1]:.3e}")

    ns_big = np.arange(2, 21, 2)
    data = {}
    for i, a in enumerate((1e-3, 1e-5)):
        p, r = _errs(a, ns_big)
        data[a] = (p, r)
        _figure(a, ns_big, p, r, f"MinimaxSqrt_repl_{i+3:02d}.png")
        print(f"a={a}: poly {p[-1]:.3e}  rat {r[-1]:.3e}")

    # a = 0: genuinely singular endpoint
    p0, r0 = _errs(0.0, ns_big)
    print(f"a=0: poly {p0[-1]:.3e}  rat {r0[-1]:.3e}")
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    p5, r5 = data[1e-5]
    ax.semilogy(ns_big, p5, 'b*-', lw=1.2)
    ax.semilogy(ns_big, r5, 'r*-', lw=1.2)
    ax.semilogy(ns_big, p0, 'bo--', lw=1.2)
    ax.semilogy(ns_big, r0, 'ro--', lw=1.2)
    ax.text(ns_big[-1] + 0.5, p5[-1], "poly a=1e-05", color='b')
    ax.text(ns_big[-1] + 0.5, r5[-1] * 2, "rat a=1e-05", color='r')
    ax.text(ns_big[-1] + 0.5, p0[-1] * 1.3, "poly a=0", color='b')
    ax.text(ns_big[-1] + 0.5, r0[-1], "rat a=0", color='r')
    ax.grid(True)
    ax.set_xlim(0, ns_big[-1] + 7)
    ax.set_title("sqrt(x) on [a,1], a = 1e-5 and 0", fontsize=12)
    ax.set_xlabel("DOF")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "MinimaxSqrt_repl_05.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fifth root: qualitatively the same
    F5 = lambda x: x ** (1.0 / 5)  # noqa: E731
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    styles = {1e-5: ('b*-', 'r*-'), 0.0: ('bo--', 'ro--')}
    for a in (1e-5, 0.0):
        perrs, rerrs = [], []
        for n in ns_big:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    perrs.append(minimax(F5, n, domain=(a, 1.0)).err)
                except Exception:
                    perrs.append(np.nan)
                try:
                    rerrs.append(minimax(F5, n // 2, rational=True,
                                         denom=n // 2,
                                         domain=(a, 1.0)).err)
                except Exception:
                    rerrs.append(np.nan)
        ps, rs_ = styles[a]
        ax.semilogy(ns_big, perrs, ps, lw=1.2)
        ax.text(ns_big[-1] + 0.2, perrs[-1], f"poly a={a:g}", color='b')
        ax.semilogy(ns_big, rerrs, rs_, lw=1.2)
        ax.text(ns_big[-1] + 0.2, rerrs[-1], f"rat a={a:g}", color='r')
        print(f"p=5 a={a}: poly {perrs[-1]:.3e}  rat {rerrs[-1]:.3e}")
    ax.grid(True)
    ax.set_xlim(0, ns_big[-1] + 7)
    ax.set_xlabel("DOF")
    ax.set_title("fifth root of x on [a,1], a = 0 and 1e-5", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "MinimaxSqrt_repl_06.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
