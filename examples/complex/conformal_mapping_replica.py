"""Conformal mapping in Chebfun.

Faithful replica of complex/ConformalMapping.m by Nick Trefethen
(October 2019): the `conformal` command applied to a smooth random
perturbation of the unit circle, with AAA rational representations of
the map and its inverse.

MATLAB's region is built with rng(0) + randnfun, which uses randn
(never bit-reproducible across MATLAB/numpy — ziggurat); our region
is therefore a different draw of the same random-function family, and
the printed accuracy numbers are for our region.

Original: https://www.chebfun.org/examples/complex/ConformalMapping.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.conformal import conformal
from chebfunjax.utils.randnfun import randnfun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ConformalMapping_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    # C = chebfun('exp(1i*pi*t)','trig') .* (1 + .15*randnfun(.2,'trig'))
    rf = randnfun(0.2, key=jax.random.PRNGKey(0))
    tt = np.linspace(-1, 1, 1201)[:-1]
    Cv = np.exp(1j * np.pi * tt) * (1 + 0.15 * np.asarray(rf(tt)))

    t0 = time.time()
    f, finv, pol, polinv = conformal(jnp.asarray(Cv))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")
    pol, polinv = np.asarray(pol), np.asarray(polinv)

    # the two-panel 'plots' figure: region and disk with grids + poles
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.6))
    ray = np.linspace(0, 1, 50)
    circ = np.exp(2j * np.pi * np.arange(0, 201) / 200)
    axL.plot(Cv.real, Cv.imag, 'b', lw=1.2)
    for th in 2 * np.pi * np.arange(1, 13) / 12:
        v = np.asarray(finv(jnp.asarray(np.exp(1j * th) * ray[1:])))
        axL.plot(v.real, v.imag, 'k', lw=0.5)
    for r in np.arange(0.1, 0.95, 0.1):
        v = np.asarray(finv(jnp.asarray(r * circ)))
        axL.plot(v.real, v.imag, 'k', lw=0.5)
    axL.plot(pol.real, pol.imag, '.r', ms=7)
    axL.set_title(f"$\\Omega$ and poles of $f$ ({len(pol)})",
                  fontsize=11)
    axR.plot(circ.real, circ.imag, 'b', lw=1.2)
    for th in 2 * np.pi * np.arange(1, 13) / 12:
        v = np.exp(1j * th) * ray
        axR.plot(v.real, v.imag, 'k', lw=0.5)
    for r in np.arange(0.1, 0.95, 0.1):
        axR.plot((r * circ).real, (r * circ).imag, 'k', lw=0.5)
    axR.plot(polinv.real, polinv.imag, '.r', ms=7)
    axR.set_title(f"$D$ and poles of $f^{{-1}}$ ({len(polinv)})",
                  fontsize=11)
    for ax in (axL, axR):
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_axis_off()
    _save(fig)

    # 10,000 uniformly distributed random points in the unit disk
    rs = np.random.RandomState(5489)
    W = (2 * rs.random_sample(20000) - 1
         + 2j * rs.random_sample(20000) - 1j)
    W = W[np.abs(W) < 1][:10000]
    t0 = time.time()
    Z = np.asarray(finv(jnp.asarray(W)))
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.plot(Cv.real, Cv.imag, 'b', lw=1.2)
    ax.plot(Z.real, Z.imag, '.k', ms=2)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    _save(fig)

    # 1000 points on the boundary curve C
    Wb = np.exp(1j * np.pi * np.linspace(-1, 1, 1001)) \
        * (1 + 0.15 * np.asarray(rf(np.linspace(-1, 1, 1001))))
    Zb = np.asarray(f(jnp.asarray(Wb)))
    print("max_deviation_from_circle =")
    print(f"     {np.max(np.abs(np.abs(Zb)-1)):.15e}")
    W2 = np.asarray(finv(jnp.asarray(Zb)))
    print("max_back_and_forth_error =")
    print(f"     {np.max(np.abs(Wb-W2)):.15e}")


if __name__ == "__main__":
    run()
