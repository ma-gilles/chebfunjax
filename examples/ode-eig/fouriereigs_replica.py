"""Periodic ODE eigenvalue problems.

Faithful replica of ode-eig/FourierEigs.m by Hadrien Montanelli
(December 2014): periodic Sturm-Liouville eigenvalue problems solved
by Fourier collocation --

    -u'' = lam u          (eigenvalues n^2, doubled for n >= 1)
    -u'' + 2q cos(2x) u = lam u    (the Mathieu equation, q = 2)

on [0, 2pi] with periodic boundary conditions.

Original: https://www.chebfun.org/examples/ode-eig/FourierEigs.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')


def _plot(V, fname):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xx = np.linspace(0, 2 * np.pi, 2000)
    for v in V:
        ax.plot(xx, np.asarray(v(xx)).real, lw=2)
    ax.set_xlim(0, 2 * np.pi)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    dom = (0.0, 2 * np.pi)

    # -u'' = lam u, periodic: eigenvalues 0, 1, 1, 4, 4.
    L = Chebop(lambda u: -u.diff(2), domain=dom)
    L.bc = "periodic"
    V, lam = L.eigs(k=5, return_eigenfunctions=True)
    lam = np.asarray(lam).real
    _plot(V, "FourierEigs_repl_01.png")

    Dexact = np.array([0.0, 1.0, 1.0, 4.0, 4.0])
    print("ans =")
    print(f"     {np.max(np.abs(np.sort(lam) - Dexact)):.15e}")

    # The eigenfunctions are periodic ...
    for v in V:
        print(v)

    # ... and satisfy the ODE to high precision.
    res = max(float((L(v) - float(el) * v).norm(np.inf))
              for el, v in zip(lam, V))
    print("ans =")
    print(f"     {res:.15e}")

    # The Mathieu equation, q = 2.
    q = 2
    L = Chebop(lambda x, u: -u.diff(2) + 2 * q * (2 * x).cos() * u,
               domain=dom)
    L.bc = "periodic"
    V, lam = L.eigs(k=5, return_eigenfunctions=True)
    lam = np.asarray(lam).real
    _plot(V, "FourierEigs_repl_02.png")

    Dwolfram = np.array([
        -1.513956885056520, -1.390676501225323, 2.379199880488686,
        3.672232706497191, 5.172665133358294])
    print("ans =")
    print(f"     {np.max(np.abs(np.sort(lam) - Dwolfram)):.15e}")

    for v in V:
        print(v)

    res = max(float((L(v) - float(el) * v).norm(np.inf))
              for el, v in zip(lam, V))
    print("ans =")
    print(f"     {res:.15e}")


if __name__ == "__main__":
    run()
