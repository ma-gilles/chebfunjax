"""Orbiting around fixed masses.

Faithful replica of ode-nonlin/Orbits.m by Nick Trefethen (May 2011):
planar orbits posed in the complex plane, integrated with ode113 —
first around a single fixed mass for several initial speeds, then
around two fixed masses, where the trajectory becomes chaotic.

Original: https://www.chebfun.org/examples/ode-nonlin/Orbits.html
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

from chebfunjax.chebfun1d.chebfun import ode113
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
U0 = -1 + 1j
TOL = 1e-10


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Orbits_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _one_mass(t, u):
    return jnp.array([u[1], -u[0] / jnp.abs(u[0])**3])


def _two_mass(t, u):
    return jnp.array([u[1],
                      -u[0] / jnp.abs(u[0])**3
                      - (u[0] - 1) / jnp.abs(u[0] - 1)**3])


def _solve(fun, tmax, v):
    uv = ode113(fun, (0.0, float(tmax)),
                jnp.array([U0, complex(v)]), rtol=TOL, atol=TOL)
    return uv[0]


def _draw(ax, u, tmax):
    t = np.linspace(0, tmax, 4000)
    z = np.asarray(u(t))
    ax.plot(z.real, z.imag, lw=1.6)
    tk = np.arange(0, tmax + 1)
    zk = np.asarray(u(jnp.asarray(tk, dtype=jnp.float64)))
    ax.plot(zk.real, zk.imag, 'k.', ms=10)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # One fixed mass, v = 1
    tmax = 23
    u = _solve(_one_mass, tmax, 1.0)
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.plot(0, 0, '.r', ms=18)
    _draw(ax, u, tmax)
    ax.set_aspect("equal")
    ax.grid(True)
    _save(fig)

    # A family of initial speeds
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.plot(0, 0, '.r', ms=18)
    for v in (0.5, 0.75, 1.0, 1.5, 2.0):
        _draw(ax, _solve(_one_mass, tmax, v), tmax)
    ax.axis([-3, 3, -3, 3])
    ax.set_aspect("equal")
    ax.grid(True)
    _save(fig)

    # Two fixed masses
    tmax = 10
    for v in (1.0, 0.9):
        uu = _solve(_two_mass, tmax, v)
        fig, ax = plt.subplots(figsize=(7.6, 6.8))
        ax.plot([0, 1], [0, 0], '.r', ms=18)
        _draw(ax, uu, tmax)
        ax.set_aspect("equal")
        ax.grid(True)
        _save(fig)

    # Diagnostics for the v = 0.9 orbit
    print("orbit_length =")
    print(f"  {float(abs(uu.diff()).sum()):.15f}")
    m = abs(uu).min()
    print("closeness =")
    print(f"   {float(m[1] if isinstance(m, tuple) else m):.15f}")


if __name__ == "__main__":
    run()
