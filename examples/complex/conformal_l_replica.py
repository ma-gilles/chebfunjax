"""Conformal mapping of an L-shaped region.

Faithful replica of complex/ConformalL.m by Nick Trefethen
(October 2019): "Schwarz-Christoffel mapping without the
Schwarz-Christoffel formula" — a least-squares harmonic expansion
with fractional powers at the reentrant corner, then AAA rational
representations of the map and its inverse, whose poles cluster
exponentially at the singularity.

Original: https://www.chebfun.org/examples/complex/ConformalL.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.aaa import aaa
from chebfunjax.utils.quadrature import chebpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')

FIG = [0]


def _chebpts_seg(n, a, b):
    x = np.asarray(chebpts(n))
    return a + (b - a) * (x + 1) / 2


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"ConformalL_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()
    N = 24
    warnings.filterwarnings("ignore")

    Z = 1j * (1 - np.tanh(12 * np.linspace(1, 0, 5 * N)))
    Z = np.concatenate([Z[:-1], _chebpts_seg(2 * N, 1j, -1 + 1j)])
    Z = np.concatenate([Z[:-1], _chebpts_seg(2 * N, -1 + 1j, -1 - 1j)])
    a = np.exp(0.25j * np.pi)
    Z = a * np.concatenate([Z / a, np.conj(Z[-2::-1] / a)])
    mZ = a**3 * Z / np.max(np.abs(Z))
    z0 = -0.5 - 0.5j
    cZ = (Z - z0) / np.max(np.abs(Z - z0))
    k1 = np.arange(0, N + 1)
    k2 = np.arange(1, N + 1)
    m2 = k2 * (2 / 3)
    m2 = np.delete(m2, np.arange(2, len(m2), 3))
    A = np.hstack([
        np.real(cZ[:, None] ** k1), np.imag(cZ[:, None] ** k2),
        np.real(mZ[:, None] ** m2), np.imag(mZ[:, None] ** m2)])
    zc = -0.2 - 0.2j
    U = -np.log(np.abs(Z - zc))
    c, *_ = np.linalg.lstsq(A, U, rcond=None)
    boundary_err = np.max(np.abs(A @ c - U))
    print("boundary_err =")
    print(f"   {boundary_err:.4e}")
    V = np.hstack([
        np.imag(cZ[:, None] ** k1), -np.real(cZ[:, None] ** k2),
        np.imag(mZ[:, None] ** m2), -np.real(mZ[:, None] ** m2)]) @ c
    W = (Z - zc) * np.exp(U + 1j * V)
    W = W / W[0]
    f, pol, *_ = aaa(jnp.asarray(W), jnp.asarray(Z),
                     tol=float(10 * boundary_err))
    finv, polinv, *_ = aaa(jnp.asarray(Z), jnp.asarray(W),
                           tol=float(10 * boundary_err))
    pol = np.asarray(pol)
    polinv = np.asarray(polinv)

    # the map, with poles of f and preimages of a polar grid
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.plot(Z.real, Z.imag, 'b', lw=1)
    ax.axis(list(1.1 * np.array([-1, 1, -1, 1])))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.plot(pol.real, pol.imag, '.r', ms=8)
    ray = np.asarray(chebpts(100))
    ray = ray[ray >= 0]
    circ = np.exp(2j * np.pi * np.arange(0, 201) / 200)
    for th in 2 * np.pi * np.arange(1, 13) / 12:
        v = np.asarray(finv(jnp.asarray(np.exp(1j * th) * ray)))
        ax.plot(v.real, v.imag, 'k', lw=0.5)
    for r in np.arange(0.1, 0.95, 0.1):
        v = np.asarray(finv(jnp.asarray(r * circ)))
        ax.plot(v.real, v.imag, 'k', lw=0.5)
    _save(fig)
    print("number_of_poles_of_f =")
    print(f"    {len(pol)}")

    # the disk, with poles of finv
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.plot(W.real, W.imag, 'b', lw=1)
    ax.axis(list(1.4 * np.array([-1, 1, -1, 1])))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.plot(polinv.real, polinv.imag, '.r', ms=8)
    for th in 2 * np.pi * np.arange(1, 13) / 12:
        v = np.exp(1j * th) * ray
        ax.plot(v.real, v.imag, 'k', lw=0.5)
    for r in np.arange(0.1, 0.95, 0.1):
        v = r * circ
        ax.plot(v.real, v.imag, 'k', lw=0.5)
    _save(fig)
    print("number_of_poles_of_finv =")
    print(f"    {len(polinv)}")

    # exponential clustering of poles at the reentrant corner
    distances = np.sort(np.abs(
        pol[(pol.real > 0) & (pol.imag > 0)]))
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.semilogy(np.arange(1, len(distances) + 1), distances, '.-',
                ms=10, lw=0.5)
    ax.grid(True)
    ax.set_title("distances of poles to singularity", fontsize=12)
    _save(fig)

    ratios = distances[1:] / distances[:-1]
    print("ratios =")
    for r in ratios:
        print(f"    {r:.4f}")

    print("total_time_for_this_example =")
    print(f"    {time.time() - t0:.4f}")


if __name__ == "__main__":
    run()
