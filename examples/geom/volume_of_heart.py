"""The volume of a heart.

Translation of geom/VolumeOfHeart.m by Rodrigo Platte
(February 2013): areas by Green's theorem, and volumes of surfaces
of revolution (torus, heart) by the divergence theorem, with surface
normals from chebfun2 partial derivatives.

Original: https://www.chebfun.org/examples/geom/VolumeOfHeart.html
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
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]


def _save(fig, close=True):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"VolumeOfHeart_{FIG[0]:02d}.png"))
    if close:
        plt.close(fig)


def _normal(x, y, z, sign=1.0):
    """Surface normal r_u x r_v (chebfun2 components; dim=2 is u)."""
    xu, xv = x.diff(dim=2), x.diff(dim=1)
    yu, yv = y.diff(dim=2), y.diff(dim=1)
    zu, zv = z.diff(dim=2), z.diff(dim=1)
    nx = (yu * zv - zu * yv) * sign
    ny = (zu * xv - xu * zv) * sign
    nz = (xu * yv - yu * xv) * sign
    return nx, ny, nz


def _quiver3(ax, fx, fy, fz, fields, dom, color, numpts=20):
    """MATLAB ``quiver3(x, y, z, F, 'numpts', n)`` for chebfun2 data.

    Arrows based at the surface points ``(fx, fy, fz)`` sampled on an
    ``numpts x numpts`` grid of the parameter domain, pointing along the
    three chebfun2 components in *fields*, autoscaled to the grid spacing.
    """
    xs = np.linspace(dom[0], dom[1], numpts)
    ys = np.linspace(dom[2], dom[3], numpts)
    XX, YY = np.meshgrid(xs, ys)
    P = [np.asarray(f(XX, YY)) for f in (fx, fy, fz)]
    W = [np.asarray(f(XX, YY)) for f in fields]
    span = max(float(np.ptp(c)) for c in P) / numpts
    wmax = float(np.max(np.sqrt(W[0]**2 + W[1]**2 + W[2]**2)))
    s = 0.9 * span / wmax if wmax > 0 else 1.0
    ax.quiver(*P, *(s * w for w in W), color=color, linewidth=1.5,
              arrow_length_ratio=0.3)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # area of an ellipse by Green's theorem
    dom = (0.0, 2 * np.pi)
    x1 = cj.chebfun(lambda t: 2 * jnp.cos(t), domain=dom)
    y1 = cj.chebfun(lambda t: jnp.sin(t), domain=dom)
    ts = np.linspace(*dom, 600)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.fill(np.asarray(x1(ts)), np.asarray(y1(ts)), 'r')
    ax.set_aspect("equal")
    _save(fig)
    A = 0.5 * float((x1 * y1.diff() - y1 * x1.diff()).sum())
    print("ans =")
    print(f"   {A:.15f}")

    # the classic heart curve (reversed orientation)
    x2 = cj.chebfun(lambda t: 16 * jnp.sin(-t)**3, domain=dom)
    y2 = cj.chebfun(lambda t: 13 * jnp.cos(-t) - 5 * jnp.cos(-2 * t)
                    - 2 * jnp.cos(-3 * t) - jnp.cos(-4 * t),
                    domain=dom)
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    tt = np.linspace(0, 2 * np.pi, 20)
    dx2 = x2.diff()
    dy2 = y2.diff()
    ax.quiver(np.asarray(x2(tt)), np.asarray(y2(tt)),
              np.asarray(dy2(tt)), -np.asarray(dx2(tt)))
    ax.fill(np.asarray(x2(ts)), np.asarray(y2(ts)), 'r')
    ax.set_aspect("equal")
    ax.set_axis_off()
    _save(fig)
    A2 = 0.5 * float((x2 * dy2 - y2 * dx2).sum())
    print("ans =")
    print(f"     {A2:.15e}")

    # torus volume and area by the divergence theorem
    d2 = (0.0, 2 * np.pi, 0.0, 2 * np.pi)
    X = cj.chebfun2(lambda u, v: (3 + jnp.cos(v)) * jnp.cos(u),
                    domain=d2)
    Y = cj.chebfun2(lambda u, v: (3 + jnp.cos(v)) * jnp.sin(u),
                    domain=d2)
    Z = cj.chebfun2(lambda u, v: jnp.sin(v) + 0 * u, domain=d2)
    nx, ny, nz = _normal(X, Y, Z)

    us = np.linspace(0, 2 * np.pi, 100)
    U, V = np.meshgrid(us, us)
    fig = plt.figure(figsize=(8.4, 6.2))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(np.asarray(X(U, V)), np.asarray(Y(U, V)),
                    np.asarray(Z(U, V)), cmap="viridis",
                    rstride=2, cstride=2)
    ax.view_init(elev=30, azim=-55)
    ax.set_box_aspect((4, 4, 1))
    _save(fig)

    # normals v (black), F = [0, 0, z] (green), surface coloured by F'*v
    Fdotv = Z * nz
    zero = 0 * Z
    fig = plt.figure(figsize=(8.4, 6.2))
    ax = fig.add_subplot(projection="3d")
    _quiver3(ax, X, Y, Z, (nx, ny, nz), d2, 'k', numpts=10)
    _quiver3(ax, X, Y, Z, (zero, zero, Z), d2, 'g', numpts=10)
    C = np.asarray(Fdotv(U, V))
    norm = matplotlib.colors.Normalize(C.min(), C.max())
    ax.plot_surface(np.asarray(X(U, V)), np.asarray(Y(U, V)),
                    np.asarray(Z(U, V)),
                    facecolors=plt.cm.viridis(norm(C)),
                    rstride=2, cstride=2, shade=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    ax.view_init(elev=64, azim=-28)
    ax.set_box_aspect((4, 4, 1))
    ax.set_axis_off()
    _save(fig)

    Vol = float(Fdotv.sum2())
    print("Vol =")
    print(f"  {Vol:.15f}")
    print("Exact =")
    print(f"  {2 * np.pi**2 * 3:.15f}")
    Area = float((nx**2 + ny**2 + nz**2).sqrt().sum2())
    print("Area =")
    print(f"     {Area:.15e}")
    print("Exact =")
    print(f"     {4 * np.pi**2 * 3:.15e}")

    # heart surface
    d3 = (0.0, 1.0, 0.0, 4 * np.pi)

    def hx(v, u):
        return jnp.sin(jnp.pi * v) * jnp.cos(u / 2)

    def hy(v, u):
        return 0.7 * jnp.sin(jnp.pi * v) * jnp.sin(u / 2)

    def hz(v, u):
        return ((v - 1) * (-49 + 50 * v + 30 * v * jnp.cos(u)
                           + jnp.cos(2 * u))
                / (-25 + jnp.cos(u)**2))

    HX = cj.chebfun2(hx, domain=d3)
    HY = cj.chebfun2(hy, domain=d3)
    HZ = cj.chebfun2(hz, domain=d3)
    nx, ny, nz = _normal(HX, HY, HZ, sign=-1.0)

    # normals, F = [0, 0, z] and the flux integrand F'*v on the heart
    zero = 0 * HZ
    vv = np.linspace(0, 1, 80)
    uu = np.linspace(0, 4 * np.pi, 160)
    VV, UU = np.meshgrid(vv, uu)
    fig = plt.figure(figsize=(8.0, 7.2))
    ax = fig.add_subplot(projection="3d")
    _quiver3(ax, HX, HY, HZ, (nx, ny, nz), d3, 'k', numpts=10)
    _quiver3(ax, HX, HY, HZ, (zero, zero, HZ), d3, 'g', numpts=10)
    C = np.asarray((HZ * nz)(VV, UU))
    norm = matplotlib.colors.Normalize(C.min(), C.max())
    ax.plot_surface(np.asarray(HX(VV, UU)), np.asarray(HY(VV, UU)),
                    np.asarray(HZ(VV, UU)),
                    facecolors=plt.cm.viridis(norm(C)),
                    rstride=2, cstride=2, shade=False)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
    ax.view_init(elev=5, azim=-45)
    ax.set_box_aspect((2, 1.4, 2.8))
    ax.set_axis_off()
    _save(fig)

    VolH = float((HZ * nz).sum2())
    print("Vol =")
    print(f"   {VolH:.15f}")
    lx = float(HX.minandmax2()[0][1]) - float(HX.minandmax2()[0][0])
    ly = float(HY.minandmax2()[0][1]) - float(HY.minandmax2()[0][0])
    lz = float(HZ.minandmax2()[0][1]) - float(HZ.minandmax2()[0][0])
    VolBox = lx * ly * lz
    print("VolBox =")
    print(f"   {VolBox:.15f}")

    # a seashell surface
    d4 = (0.0, 6 * np.pi, 0.0, 2 * np.pi)
    SX = cj.chebfun2(
        lambda u, v: 2 * (1 - jnp.exp(u / (6 * jnp.pi)))
        * jnp.cos(u) * jnp.cos(v / 2)**2, domain=d4)
    SY = cj.chebfun2(
        lambda u, v: 2 * (-1 + jnp.exp(u / (6 * jnp.pi)))
        * jnp.sin(u) * jnp.cos(v / 2)**2, domain=d4)
    SZ = cj.chebfun2(
        lambda u, v: 1 - jnp.exp(u / (3 * jnp.pi)) - jnp.sin(v)
        + jnp.exp(u / (6 * jnp.pi)) * jnp.sin(v), domain=d4)

    uu2 = np.linspace(0, 6 * np.pi, 240)
    vv2 = np.linspace(0, 2 * np.pi, 100)
    UU2, VV2 = np.meshgrid(uu2, vv2)
    fig = plt.figure(figsize=(8.4, 6.6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(np.asarray(SX(UU2, VV2)),
                    np.asarray(SY(UU2, VV2)),
                    np.asarray(SZ(UU2, VV2)), cmap="viridis",
                    rstride=2, cstride=2)
    ax.view_init(elev=10, azim=160)
    ax.set_box_aspect((1, 1, 1))
    _save(fig, close=False)

    # hold on: F = [0, 0, z] is parallel to the open part of the shell
    zero = 0 * SZ
    _quiver3(ax, SX, SY, SZ, (zero, zero, SZ), d4, 'g')
    _save(fig)

    nx, ny, nz = _normal(SX, SY, SZ, sign=-1.0)
    VolS = float((SZ * nz).sum2())
    print("Vol =")
    print(f"  {VolS:.15f}")
    lx = (float(SX.minandmax2()[0][1])
          - float(SX.minandmax2()[0][0]))
    ly = (float(SY.minandmax2()[0][1])
          - float(SY.minandmax2()[0][0]))
    lz = (float(SZ.minandmax2()[0][1])
          - float(SZ.minandmax2()[0][0]))
    VolBoxS = lx * ly * lz
    print("VolBox =")
    print(f"     {VolBoxS:.15e}")
    print("ans =")
    print(f"   {VolS / VolBoxS:.15f}")


if __name__ == "__main__":
    run()
