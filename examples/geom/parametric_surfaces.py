"""Parametric surfaces.

Faithful replica of geom/ParametricSurfaces.m by Rodrigo Platte
(March 2013): a gallery of surfaces built from chebfun2 coordinate
functions — cones, spheres, a seashell, a Mobius strip with its
normal and tangent frames, and a Klein bottle.

Original: https://www.chebfun.org/examples/geom/ParametricSurfaces.html
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

from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]


def _surf(x, y, z, c=None, view=None, aspect=None, cmap="viridis"):
    FIG[0] += 1
    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(projection="3d")
    kw = dict(rstride=2, cstride=2, linewidth=0)
    if c is None:
        ax.plot_surface(x, y, z, cmap=cmap, **kw)
    else:
        norm = plt.Normalize(c.min(), c.max())
        ax.plot_surface(x, y, z,
                        facecolors=plt.get_cmap(cmap)(norm(c)),
                        **kw)
    if view:
        ax.view_init(elev=view[1], azim=view[0])
    if aspect:
        ax.set_box_aspect(aspect)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ParametricSurfaces_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # cone and hyperboloid
    u = np.linspace(-1, 1, 80)
    v = np.linspace(0, 2 * np.pi, 160)
    U, V = np.meshgrid(u, v)
    _surf(U * np.cos(V), U * np.sin(V), U)
    R = np.sqrt(0.25 + U**2)
    _surf(R * np.cos(V), R * np.sin(V), U)
    _surf(R * np.cos(V), R * np.sin(V), U,
          c=(1 - U) * np.sin(10 * V))

    # sphere, bumpy sphere, colored sphere
    t = np.linspace(0, 2 * np.pi, 160)
    p = np.linspace(-np.pi / 2, np.pi / 2, 80)
    T, P = np.meshgrid(t, p)
    X = np.cos(P) * np.cos(T)
    Y = np.cos(P) * np.sin(T)
    Z = np.sin(P)
    _surf(X, Y, Z)
    delta = (np.sin(20 * X) + np.sin(20 * Y) + np.sin(20 * Z))
    R2 = 1 + 0.05 * delta
    _surf(R2 * np.cos(P) * np.cos(T), R2 * np.cos(P) * np.sin(T),
          R2 * np.sin(P))
    _surf(R2 * np.cos(P) * np.cos(T), R2 * np.cos(P) * np.sin(T),
          R2 * np.sin(P), c=np.sin(10 * T) * np.cos(15 * P),
          view=(0, 90))

    # seashell
    su = np.linspace(0, 6 * np.pi, 240)
    sv = np.linspace(0, 2 * np.pi, 100)
    SU, SV = np.meshgrid(su, sv)
    SX = 2 * (1 - np.exp(SU / (6 * np.pi))) * np.cos(SU) \
        * np.cos(SV / 2)**2
    SY = 2 * (-1 + np.exp(SU / (6 * np.pi))) * np.sin(SU) \
        * np.cos(SV / 2)**2
    SZ = (1 - np.exp(SU / (3 * np.pi)) - np.sin(SV)
          + np.exp(SU / (6 * np.pi)) * np.sin(SV))
    _surf(SX, SY, SZ, view=(160, 10))

    # Mobius strip with orthogonality check of the tangent frame
    mu = np.linspace(0, 2 * np.pi, 200)
    mv = np.linspace(-1, 1, 60)
    MU, MV = np.meshgrid(mu, mv)
    MX = (1 + 0.5 * MV * np.cos(MU / 2)) * np.cos(MU)
    MY = (1 + 0.5 * MV * np.cos(MU / 2)) * np.sin(MU)
    MZ = 0.5 * MV * np.sin(MU / 2)
    _surf(MX, MY, MZ, view=(-80, 65))

    # analytic tangent vectors r_u, r_v and their inner product
    ru = np.stack([
        -0.25 * MV * np.sin(MU / 2) * np.cos(MU)
        - (1 + 0.5 * MV * np.cos(MU / 2)) * np.sin(MU),
        -0.25 * MV * np.sin(MU / 2) * np.sin(MU)
        + (1 + 0.5 * MV * np.cos(MU / 2)) * np.cos(MU),
        0.25 * MV * np.cos(MU / 2)])
    rv = np.stack([0.5 * np.cos(MU / 2) * np.cos(MU),
                   0.5 * np.cos(MU / 2) * np.sin(MU),
                   0.5 * np.sin(MU / 2)])
    dot = np.sum(ru * rv, axis=0)
    print("ans =")
    print(f"     {np.max(np.abs(dot)):.15e}")

    # Klein bottle
    ku = np.linspace(0, 2 * np.pi, 200)
    kv = np.linspace(0, 2 * np.pi, 200)
    KU, KV = np.meshgrid(ku, kv)
    r = 3
    KX = (r + np.cos(KU / 2) * np.sin(KV)
          - np.sin(KU / 2) * np.sin(2 * KV)) * np.cos(KU)
    KY = (r + np.cos(KU / 2) * np.sin(KV)
          - np.sin(KU / 2) * np.sin(2 * KV)) * np.sin(KU)
    KZ = (np.sin(KU / 2) * np.sin(KV)
          + np.cos(KU / 2) * np.sin(2 * KV))
    _surf(KX, KY, KZ, view=(0, 90), cmap="hot")


if __name__ == "__main__":
    run()
