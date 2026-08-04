"""Rectangular spectral discretizations.

Faithful replica of ode-linear/SpectralDisc.m by Nick Trefethen
(April 2016): solving the Airy-like problem u'' - xu = 0 on
[-20, 10] with the side conditions int(u) = 1 and u(10) = u(-20),
first with explicit rectangular differentiation matrices
(diffmat([n n+2]), introw, diffrow), then with chebop, and finally
displaying the small rectangular system matrices for n = 1..4.

Original: https://www.chebfun.org/examples/ode-linear/SpectralDisc.html
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

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.diffmat import diffmat, diffrow, introw
from chebfunjax.utils.misc import gridsample
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
X = (-20.0, 10.0)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"SpectralDisc_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _assemble(n):
    L = (np.asarray(diffmat((n, n + 2), 2, domain=X))
         - np.diag(np.asarray(gridsample(lambda x: x, n, X)))
         @ np.asarray(diffmat((n, n + 2), 0, domain=X)))
    vT = np.asarray(introw(n + 2, domain=X))
    wT = (np.asarray(diffrow(n + 2, 0, X[1], domain=X))
          - np.asarray(diffrow(n + 2, 0, X[0], domain=X)))
    return np.vstack([L, vT, wT])


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Explicit rectangular matrices, n = 85
    n = 85
    A = _assemble(n)
    rhs = np.concatenate([np.zeros(n), [1.0], [0.0]])
    uvals = np.linalg.solve(A, rhs)
    ufun = Chebfun.from_values(uvals, Domain(X))
    pts = np.asarray(chebpts(n + 2))
    xpts = 0.5 * (X[1] - X[0]) * pts + 0.5 * (X[0] + X[1])
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    t = np.linspace(*X, 1500)
    ax.plot(t, np.asarray(ufun(t)), lw=1.2)
    ax.plot(xpts, uvals, '.')
    ax.grid(True)
    _save(fig)

    # The same problem via chebop
    L = Chebop(lambda x, u: u.diff(2) - x * u, domain=X)
    L.bc = lambda x, u: [u.sum() - 1, u(10.0) - u(-20.0)]
    u = L.solve(0.0)
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.plot(t, np.asarray(u(t)), lw=1.2)
    ax.grid(True)
    _save(fig)
    print("ans =")
    print(f"    {len(u)}")

    # Small rectangular system matrices
    for nn in range(1, 5):
        An = _assemble(nn)
        print("A =")
        for row in An:
            print("  " + "".join(f"{v:10.4f}" for v in row))


if __name__ == "__main__":
    run()
