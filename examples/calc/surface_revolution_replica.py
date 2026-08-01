"""Surfaces of revolution.

Faithful replica of calc/SurfaceRevolution.m by Georges Klein: surfaces
of revolution from chebfuns, with volume, surface area, center of
gravity, and moment of inertia computed as chebfun integrals.

Original: https://www.chebfun.org/examples/calc/SurfaceRevolution.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'calc')


def _cylinder_surf(f, dom, stem, orient="z", zscale=1.0, view=None):
    ts = np.linspace(dom[0], dom[1], 120)
    th = np.linspace(0, 2 * np.pi, 80)
    R = np.asarray(f(jnp.asarray(ts)))
    TH, T = np.meshgrid(th, ts)
    RR = np.tile(R[:, None], (1, len(th)))
    X, Y = RR * np.cos(TH), RR * np.sin(TH)
    Z = np.tile(((ts - dom[0]) / (dom[1] - dom[0]))[:, None],
                (1, len(th))) * zscale
    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(projection="3d")
    if orient == "z":
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85,
                        linewidth=0)
    else:
        ax.plot_surface(Z * (dom[1] - dom[0]) + dom[0], np.flipud(Y), X,
                        cmap="viridis", alpha=0.85, linewidth=0)
    if view:
        ax.view_init(*view)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, stem + ".png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)

    f = cj.chebfun(lambda x: jnp.sqrt(1.0001 - x ** 2))
    _cylinder_surf(f, (-1, 1), "SurfaceRevolution_repl_01")
    f = cj.chebfun(lambda x: -3 * x + 3)
    _cylinder_surf(f, (-1, 1), "SurfaceRevolution_repl_02")
    f = cj.chebfun(lambda x: 2.8 * jnp.sin(0.2 * x - 0.1) + 6.3,
                   domain=[-5, 35])
    _cylinder_surf(f, (-5, 35), "SurfaceRevolution_repl_03",
                   orient="x")

    x = cj.chebfun(lambda t: t, domain=[0, 2 * np.pi])
    f = cj.chebfun(lambda t: jnp.sqrt(4 + 2 * jnp.sin(2 * t)),
                   domain=[0, 2 * np.pi])
    _cylinder_surf(f, (0, 2 * np.pi), "SurfaceRevolution_repl_04",
                   zscale=8.0)

    V = np.pi * float((f * f).sum())
    print("V =")
    print(f"  {V:.15f}")
    print("error =")
    print(f"    {V - 8 * np.pi ** 2:.15e}")
    A = 2 * np.pi * float((f * (1 + abs(f.diff()) ** 2).sqrt()).sum())
    print("A =")
    print(f"  {A:.15f}")
    zG = np.pi / V * float((x * f * f).sum())
    print("zG =")
    print(f"   {zG:.15f}")
    J = np.pi / 2 * float((f * f * f * f).sum())
    print("J =")
    print(f"     {J:.15e}")

    fls = cj.chebfun(lambda t: 1.0 / (1 + 8 * t ** 2))
    _cylinder_surf(fls, (-1, 1), "SurfaceRevolution_repl_05",
                   zscale=0.25, view=(10, 0))
    return True


if __name__ == "__main__":
    run()
