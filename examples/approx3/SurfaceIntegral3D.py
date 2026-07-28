"""Integration of scalar functions over 2D surfaces in 3D.

Faithful port of approx3/SurfaceIntegral3D.m by Behnam Hashemi (June 2016).
Computes surface integrals int_S f dS over parametric surfaces using the
chebfun3 ``integral2(f, S)`` operator, where S is a parametric surface
(x(u,v), y(u,v), z(u,v)) and the area element |S_u x S_v| is formed
automatically: a unit sphere, a cone, two seashells, and a spring.

Original: https://www.chebfun.org/examples/approx3/SurfaceIntegral3D.html
Copyright 2016 by The University of Oxford and The Chebfun Developers.

Output-parity note (measured): all five published surface integrals are
reproduced to ~13-15 significant figures using the library ``integral2``
(sphere 4pi/3, cone 14pi/3, seashell 6030.79, another seashell -2.984e7,
spring 1878.45), replacing the prior hand-rolled trapezoid quadrature (which
was only ~1% accurate).  The "another seashell" case also fixes a real
operator-precedence port bug: MATLAB's ``((u+3)/8*pi).^2`` is
``((u+3)/8*pi)^2`` (left-to-right), not ``((u+3)/(8*pi))^2`` -- the old port
integrated the wrong surface and was off by ~10x.
"""
import matplotlib

matplotlib.use("Agg")
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.plotting import PARULA, _setup_3d_axes, chebfun_style

chebfun_style()

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)), "docs", "images", "approx3"
)
os.makedirs(_IMG_DIR, exist_ok=True)

_PI = float(np.pi)


def run():
    # ------------------------------------------------------------------
    # Example 1: int_{unit sphere} x^2 dS = 4*pi/3.
    # ------------------------------------------------------------------
    f = chebfun3(lambda x, y, z: x**2)
    S = lambda u, v: (jnp.sin(u) * jnp.cos(v), jnp.sin(u) * jnp.sin(v),
                      jnp.cos(u))
    I = float(f.integral2(S, domain=(0, _PI, 0, 2 * _PI)))
    print("I =")
    print(f"   {I:.15f}")
    print("error =")
    print(f"     {abs(I - 4 * _PI / 3):.15e}")

    # ------------------------------------------------------------------
    # Example 2: sqrt(1+x^2+y^2) over a cone; exact = 14*pi/3.
    # ------------------------------------------------------------------
    f = chebfun3(lambda x, y, z: jnp.sqrt(1 + x**2 + y**2),
                 domain=(-3, 3, -3, 3, -3, 3))
    S = lambda u, v: (u * jnp.cos(v), u * jnp.sin(v), v)
    I = float(f.integral2(S, domain=(0, 2, 0, _PI)))
    print("I =")
    print(f"  {I:.15f}")
    print("exact =")
    print(f"  {14 * _PI / 3:.15f}")

    # ------------------------------------------------------------------
    # Example 3: x+y+z over a seashell surface.
    # ------------------------------------------------------------------
    f = chebfun3(lambda x, y, z: x + y + z, domain=(-6, 6, -6, 6, 0, 25))

    def S3(u, v):
        a = 5 / 4 * (1 - v / (2 * _PI))
        return (a * jnp.cos(2 * v) * (1 + jnp.cos(u)) + jnp.cos(2 * v),
                a * jnp.sin(2 * v) * (1 + jnp.cos(u)) + jnp.sin(2 * v),
                10 * v / (2 * _PI) + a * jnp.sin(u) + 15)

    I = float(f.integral2(S3, domain=(0, 2 * _PI, -2 * _PI, 2 * _PI)))
    print("I =")
    print(f"     {I:.15e}")

    # ------------------------------------------------------------------
    # Example 4: x+y+z over another seashell.
    # ------------------------------------------------------------------
    f = chebfun3(lambda x, y, z: x + y + z,
                 domain=(-100, 100, -100, 100, -400, 0))

    def S4(u, v):
        return (u * jnp.cos(u) * (jnp.cos(v) + 1),
                u * jnp.sin(u) * (jnp.cos(v) + 1),
                u * jnp.sin(v) - ((u + 3) / 8 * _PI)**2 - 20)

    I = float(f.integral2(S4, domain=(0, 13 * _PI, -_PI, _PI)))
    print("I =")
    print(f"    {I:.15e}")

    # ------------------------------------------------------------------
    # Example 5: x+y+z over a spring.
    # ------------------------------------------------------------------
    r1, r2, tc = 0.5, 0.5, 1.5
    f = chebfun3(lambda x, y, z: x + y + z, domain=(-2, 2, -2, 2, -2, 10))

    def S5(u, v):
        return ((1 - r1 * jnp.cos(v)) * jnp.cos(u),
                (1 - r1 * jnp.cos(v)) * jnp.sin(u),
                r2 * (jnp.sin(v) + tc * u / _PI))

    I = float(f.integral2(S5, domain=(0, 10 * _PI, 0, 10 * _PI)))
    exact = 1878.4483067846025004401820388947
    print("I =")
    print(f"     {I:.15e}")
    print("error =")
    print(f"     {(I - exact) / exact:.15e}")

    # ------------------------------------------------------------------
    # Plot: the four surfaces.
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 3.5))

    ax1 = fig.add_subplot(141, projection="3d")
    _setup_3d_axes(ax1, fig)
    us = np.linspace(0, np.pi, 50)
    vs = np.linspace(0, 2 * np.pi, 80)
    U, V = np.meshgrid(us, vs)
    ax1.plot_surface(np.sin(U) * np.cos(V), np.sin(U) * np.sin(V), np.cos(U),
                     cmap=PARULA, linewidth=0, antialiased=True)
    ax1.set_title("unit sphere", fontsize=9, pad=0)

    ax2 = fig.add_subplot(142, projection="3d")
    _setup_3d_axes(ax2, fig)
    ush = np.linspace(0, 2 * np.pi, 60)
    vsh = np.linspace(-2 * np.pi, 2 * np.pi, 90)
    U, V = np.meshgrid(ush, vsh)
    a = 5 / 4 * (1 - V / (2 * np.pi))
    ax2.plot_surface(a * np.cos(2 * V) * (1 + np.cos(U)) + np.cos(2 * V),
                     a * np.sin(2 * V) * (1 + np.cos(U)) + np.sin(2 * V),
                     10 * V / (2 * np.pi) + a * np.sin(U) + 15,
                     cmap=PARULA, linewidth=0, antialiased=True)
    ax2.set_title("seashell", fontsize=9, pad=0)

    ax3 = fig.add_subplot(143, projection="3d")
    _setup_3d_axes(ax3, fig)
    u2 = np.linspace(0, 13 * np.pi, 90)
    v2 = np.linspace(-np.pi, np.pi, 60)
    U, V = np.meshgrid(u2, v2)
    ax3.plot_surface(U * np.cos(U) * (np.cos(V) + 1),
                     U * np.sin(U) * (np.cos(V) + 1),
                     U * np.sin(V) - ((U + 3) / 8 * np.pi)**2 - 20,
                     cmap=PARULA, linewidth=0, antialiased=True)
    ax3.set_title("another seashell", fontsize=9, pad=0)

    ax4 = fig.add_subplot(144, projection="3d")
    _setup_3d_axes(ax4, fig)
    usp = np.linspace(0, 10 * np.pi, 100)
    vsp = np.linspace(0, 10 * np.pi, 60)
    U, V = np.meshgrid(usp, vsp)
    ax4.plot_surface((1 - r1 * np.cos(V)) * np.cos(U),
                     (1 - r1 * np.cos(V)) * np.sin(U),
                     r2 * (np.sin(V) + tc * U / np.pi),
                     cmap=PARULA, linewidth=0, antialiased=True)
    ax4.set_title("spring", fontsize=9, pad=0)

    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG_DIR, "SurfaceIntegral3D.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    return True


if __name__ == "__main__":
    run()
