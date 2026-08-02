"""Minimax approximation in the complex plane.

Faithful replica of complex/ComplexMinimax.m by Nick Trefethen
(December 2019): AAA-Lawson near-minimax rational approximation of
exp(z) on a disk, a triangle, and a 'crazy' random domain, with
near-circular error curves of winding number 2n+1.

Original: https://www.chebfun.org/examples/complex/ComplexMinimax.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

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


def _error_plot(E, ylim_val, title, dots=False, err=None):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    if err is not None:
        th = np.linspace(0, 2 * np.pi, 400)
        circ = err * np.exp(1j * th)
        ax.plot(circ.real, circ.imag, 'r', lw=1.2)
    if dots:
        ax.plot(E.real, E.imag, '.k', ms=5)
    else:
        ax.plot(E.real, E.imag, lw=1.0)
    ax.grid(True)
    ax.set_ylim(-ylim_val, ylim_val)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ComplexMinimax_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()

    # exp(z) on the unit circle, near-minimax degree 4
    Z = np.exp(2j * np.pi * np.arange(1, 1001) / 1000)
    F = np.exp(Z)
    r, *_ = aaa(jnp.asarray(F), jnp.asarray(Z), degree=4)
    E = F - np.asarray(r(Z))
    err = np.max(np.abs(E))
    print("error =")
    print(f"     {err:.15e}")
    _error_plot(E, 5e-8, "degree 4 error curve on a disk")
    a = np.unwrap(np.angle(np.concatenate([E, E[:1]])))
    print("winding_number =")
    print(f"     {round((a[-1]-a[0])/(2*np.pi))}")

    # ... on a triangle with corners at the cube roots of unity
    omega = np.exp(2j * np.pi / 3)
    edge = np.asarray(chebpts(1001))
    Zedge = (1 + edge) / 2 * (omega - 1) + 1     # side 1 -> omega
    Z = np.concatenate([Zedge, omega * Zedge, omega**2 * Zedge])
    F = np.exp(Z)
    r, *_ = aaa(jnp.asarray(F), jnp.asarray(Z), degree=4)
    E = F - np.asarray(r(Z))
    err = np.max(np.abs(E))
    print("error =")
    print(f"     {err:.15e}")
    _error_plot(E, 3.5e-9, "degree 4 error curve on a triangle")
    a = np.unwrap(np.angle(np.concatenate([E, E[:1]])))
    print("winding_number =")
    print(f"     {round((a[-1]-a[0])/(2*np.pi))}")

    # a 'crazy' domain: half-disk of random points plus an interval
    rs = np.random.RandomState(5489)
    Zc = (0.5 + rs.random_sample(2000)) + 1j * (0.5
                                                + rs.random_sample(2000))
    Zc = Zc[np.abs(Zc - (1 + 1j)) < 0.5][:1000]
    Z = np.concatenate([Zc, np.asarray(chebpts(500))])
    FIG_dom = FIG[0] + 1
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(Z.real, Z.imag, '.k', ms=4)
    ax.set_ylim(-0.5, 2)
    ax.set_aspect("equal")
    ax.set_title("A crazy domain for approximation", fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    FIG[0] += 1
    fig.savefig(os.path.join(
        _IMG, f"ComplexMinimax_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)

    F = np.exp(Z)
    r, *_ = aaa(jnp.asarray(F), jnp.asarray(Z), degree=4)
    E = F - np.asarray(r(Z))
    err = np.max(np.abs(E))
    print("error =")
    print(f"     {err:.15e}")
    _error_plot(E, 4e-8, "Error on the crazy domain", dots=True,
                err=err)

    r, *_ = aaa(jnp.asarray(F), jnp.asarray(Z), degree=4, lawson=200)
    E = F - np.asarray(r(Z))
    err = np.max(np.abs(E))
    print("error =")
    print(f"     {err:.15e}")
    _error_plot(E, 4e-8, "Error with 200 Lawson steps", dots=True,
                err=err)
    print("total_time_for_this_example =")
    print(f"   {time.time()-t0:.6f}")


if __name__ == "__main__":
    run()
