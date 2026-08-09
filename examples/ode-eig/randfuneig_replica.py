"""Eigenvalues of random operators.

Faithful replica of ode-eig/Randfuneig.m by Yuji Nakatsukasa (April
2017): the circular law for random matrices, eigenvalues of random
low-rank products B'A, and the continuous analogue -- eigenvalues of
the Fredholm integral operator whose kernel is a random bivariate
function (randnfun2), including a variant with max-norm-bounded
coefficient support.

Random draws use numpy's generator: MATLAB's randn stream cannot be
reproduced, and these are statistical illustrations -- each figure
shows one sample of the same law as the published one.

Original: https://www.chebfun.org/examples/ode-eig/Randfuneig.html
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

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.random import randnfun2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')
FIG = [0]


def _plot_eigs(ei):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ei = np.asarray(ei)
    ax.plot(ei.real, ei.imag, 'k.', markersize=4, linestyle='none')
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), lw=1.6)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"Randfuneig_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(2017)

    # 1. Circular law for a Gaussian random matrix.
    n = 1000
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    _plot_eigs(np.linalg.eigvals(A))

    # 2. Random low-rank products, aspect ratio 10.
    m = 10 * n
    A = rng.standard_normal((m, n)) / (m * n) ** 0.25
    B = rng.standard_normal((m, n)) / (m * n) ** 0.25
    _plot_eigs(np.linalg.eigvals(B.T @ A))

    # 3. Product of two square random matrices: clustered at the origin.
    m = n
    A = rng.standard_normal((m, n)) / (m * n) ** 0.25
    B = rng.standard_normal((m, n)) / (m * n) ** 0.25
    _plot_eigs(np.linalg.eigvals(B.T @ A))

    # 4. Fredholm eigenvalues of a random bivariate kernel.
    dt = 0.01
    f = randnfun2(dt, (-1, 1, -1, 1), seed=2017, big=True)
    ei = np.asarray(f.eig())
    print(f"Number of nonzero eigenvalues: {len(ei)}")
    _plot_eigs(ei)

    # 5. Same, with max-norm-bounded coefficient support (the essence of
    # randnfun2 with a square instead of a disc of wavenumbers).
    nn = round(2 / dt)
    c = (rng.standard_normal((2 * nn + 1, 2 * nn + 1))
         + 1j * rng.standard_normal((2 * nn + 1, 2 * nn + 1)))
    kx, ky = np.meshgrid(np.arange(-nn, nn + 1), np.arange(-nn, nn + 1))
    keep = np.maximum(np.abs(kx / nn), np.abs(ky / nn)) <= 1  # max-norm
    c = c * keep
    c = c / np.sqrt(np.count_nonzero(c))
    nbig = round(1.2 * 2 / dt + 2)
    dom2 = (-1.0, -1.0 + nbig * dt, -1.0, -1.0 + nbig * dt)
    Lx = dom2[1] - dom2[0]

    def fser(x, y):
        # Random periodic Fourier series with the square support,
        # evaluated separably on the sample grid.
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        shp = x.shape
        xf = x.ravel()
        yf = y.ravel()
        Ex = np.exp(2j * np.pi * np.outer(xf - dom2[0],
                                          np.arange(-nn, nn + 1)) / Lx)
        Ey = np.exp(2j * np.pi * np.outer(yf - dom2[2],
                                          np.arange(-nn, nn + 1)) / Lx)
        out = np.einsum('pk,kl,pl->p', Ex, c.T, Ey).real
        return jnp.asarray(out.reshape(shp), dtype=jnp.float64)

    f = Chebfun2.from_function(fser, domain=dom2)
    f = f.restrict((-1, 1, -1, 1)) * (1 / np.sqrt(dt))
    ei = np.asarray(f.eig())
    print(f"Number of nonzero eigenvalues: {len(ei)}")
    _plot_eigs(ei)


if __name__ == "__main__":
    run()
