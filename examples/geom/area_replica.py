"""Areas and centroids of planar regions.

Faithful replica of geom/Area.m by Stefan Guettel (October 2011):
the area enclosed by parametrized curves via Green's theorem, and
the centroid of a region, all as chebfun integrals.

Original: https://www.chebfun.org/examples/geom/Area.html
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

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"Area_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    dom = (0.0, 2 * np.pi)
    b, m = 1, 7
    a = (m - 1) * b
    x = cj.chebfun(lambda t: (a + b) * jnp.cos(t)
                   - b * jnp.cos((a + b) / b * t), domain=dom)
    y = cj.chebfun(lambda t: (a + b) * jnp.sin(t)
                   - b * jnp.sin((a + b) / b * t), domain=dom)
    ts = np.linspace(*dom, 1200)
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    ax.fill(np.asarray(x(ts)), np.asarray(y(ts)),
            color=(0.6, 0.6, 1))
    ax.set_aspect("equal")
    _save(fig)

    print(f"epicycloid x: length {len(x)}, y: length {len(y)}")
    A = float((x * y.diff()).sum())
    print("A =")
    print(f"     {A:.15e}")
    print("exact =")
    print(f"     {np.pi * b**2 * (m**2 + m):.15e}")

    z = cj.chebfun(lambda t: jnp.exp(1j * t)
                   + (1 + 1j) * jnp.sin(6 * t)**2, domain=dom)
    zr = z.real()
    zi = z.imag()
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    zv = np.asarray(z(ts))
    ax.fill(zv.real, zv.imag, color=(0.6, 1, 0.6))
    ax.set_aspect("equal")
    A2 = float((zr * zi.diff()).sum())
    print("ans =")
    print(f"   {A2:.15f}")
    print(f"   {np.pi:.15f}")

    c = complex(np.asarray((z.diff() * z * z.conj()).sum())) \
        / (2j * A2)
    print(f"centroid = {c.real:.6f} + {c.imag:.6f}i")
    ax.plot(c.real, c.imag, 'r+', ms=16, mew=2)
    _save(fig)


if __name__ == "__main__":
    run()
