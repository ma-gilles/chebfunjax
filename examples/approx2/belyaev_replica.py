"""2D zero set example of Dmitry Belyaev.

Faithful replica of approx2/Belyaev.m (Trefethen, 2019): zero sets of
a random combination of four plane waves at wavenumbers k = 8, 16, 32,
computed with Chebfun2 roots, with per-component arc lengths.

The random coefficients `a` are the EXACT values MATLAB's rng(1)
stream produces (dumped from MATLAB R2025b), so the zero sets are the
same curves as the published example.

Original: https://www.chebfun.org/examples/approx2/Belyaev.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2')

# MATLAB: rng(1); a = randn(1,4) + 1i*rand(1,4)  (dumped from R2025b)
A = np.array([
    -0.64901376519124065 + 0.1862602113776709j,
    1.1811660419655317 + 0.34556072704304774j,
    -0.7584532972836916 + 0.39676747423066994j,
    -1.1096130385015222 + 0.53881673400335695j,
])


def wave(k):
    def f(x, y):
        return np.real(
            A[0] * np.exp(1j * np.pi * (k * x - y))
            + A[1] * np.exp(1j * np.pi * (k * x + y))
            + A[2] * np.exp(1j * np.pi * (k * y - x))
            + A[3] * np.exp(1j * np.pi * (k * y + x)))
    return f


def arclength(c, nq=2000):
    """norm(diff(f), 1) for a complex parametrized curve chebfun."""
    dc = c.diff()
    a, b = float(c.domain.a), float(c.domain.b)
    t, w = np.polynomial.legendre.leggauss(nq)
    t = 0.5 * (b - a) * t + 0.5 * (b + a)
    v = np.asarray(dc(t))
    return float(0.5 * (b - a) * np.sum(w * np.abs(v)))


def _plot(curves, lw, fname):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for c in curves:
        t = np.linspace(float(c.domain.a), float(c.domain.b), 800)
        z = np.asarray(c(t))
        ax.plot(np.real(z), np.imag(z), 'b', lw=lw)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"number of components: {len(curves)}")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    f8 = Chebfun2.from_function(wave(8))
    r = f8.roots()
    _plot(r, 2, "Belyaev_repl_01.png")

    print("ans =")
    print(f"   Inf    {len(r)}")

    al = sorted(arclength(c) for c in r)
    print("ans =")
    for v in al:
        print(f"   {v:.15f}")

    r16 = Chebfun2.from_function(wave(16)).roots()
    _plot(r16, 1.2, "Belyaev_repl_02.png")

    r32 = Chebfun2.from_function(wave(32)).roots()
    _plot(r32, 0.7, "Belyaev_repl_03.png")

    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")


if __name__ == "__main__":
    run()
