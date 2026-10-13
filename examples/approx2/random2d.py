"""Random functions in 2D.

Translation of approx2/Random2D.m (Trefethen, 2017): smooth
random functions from randnfun2 at space scales lambda = 0.2, 0.1,
0.05 on a 2x1 rectangle -- zebra plots, contours, high-accuracy zero
contours from roots, a 3D view, and the periodic variant.

Random draws use seeded JAX PRNG keys. Their values do not currently
reproduce MATLAB's RNG stream: MATLAB seeds once with rng(0) and advances
one stream, while this translation resets separate seeds for each panel.
Matching MATLAB's stream remains a parity task.

Original: https://www.chebfun.org/examples/approx2/Random2D.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax._surface_lighting import (
    matlab_camera_and_camlight_left,
    matlab_flat_face_lighting,
)
from chebfunjax.plotting import PARULA, chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.random import randnfun2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.environ.get(
    "CHEBFUNJAX_IMAGE_DIR",
    os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx2'),
)

DOM = (0.0, 2.0, 0.0, 1.0)


def _grid(n=400):
    gx = np.linspace(0, 2, 2 * n)
    gy = np.linspace(0, 1, n)
    return np.meshgrid(gx, gy)


def _zebra(f, k):
    X, Y = _grid()
    Z = np.asarray(f(X, Y))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.contourf(X, Y, Z, [float(Z.min()) - 1, 0, float(Z.max()) + 1],
                colors=[(0, 0, 0), (1, 1, 1)])
    ax.set_aspect("equal")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 2.01, .5))
    ax.set_yticks(np.arange(0, 1.01, .5))
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Random2D_{k:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    lam = 0.2
    f = randnfun2(lam, DOM, seed=0)
    _zebra(f, 1)

    # contour plot
    X, Y = _grid()
    Z = np.asarray(f(X, Y))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.contour(X, Y, Z, 10)
    ax.set_aspect("equal")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 2.01, .5))
    ax.set_yticks(np.arange(0, 1.01, .5))
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Random2D_02.png"))
    plt.close(fig)

    # zero contours via roots
    c = f.roots()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for cc in c:
        t = np.linspace(float(cc.domain.a), float(cc.domain.b), 500)
        z = np.asarray(cc(t))
        ax.plot(np.real(z), np.imag(z), 'b', lw=1.0)
    ax.set_aspect("equal")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 2.01, .5))
    ax.set_yticks(np.arange(0, 1.01, .5))
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Random2D_03.png"))
    plt.close(fig)
    print(f"zero-contour components: {len(c)}")

    # 3D surface: Chebfun surf(f) requests interpolated colors, no edges,
    # then MATLAB's default colormap and a local camlight left.
    fig = plt.figure(figsize=(9.0, 5.4))
    ax = fig.add_subplot(projection="3d")
    # @separableApprox/surf.m samples each domain axis at minPlotNum=200.
    surface_x = np.linspace(0.0, 2.0, 200)
    surface_y = np.linspace(0.0, 1.0, 200)
    xs, ys = np.meshgrid(surface_x, surface_y)
    zs = np.asarray(f(xs, ys))
    zmin, zmax = float(np.min(zs)), float(np.max(zs))
    norm = Normalize(vmin=zmin, vmax=zmax if zmax > zmin else zmin + 1.0)
    vertex_rgba = PARULA(norm(zs))
    vertices = np.stack((xs, ys, zs), axis=-1)
    quads = np.stack((vertices[:-1, :-1], vertices[:-1, 1:],
                      vertices[1:, 1:], vertices[1:, :-1]), axis=2)
    ax.auto_scale_xyz(xs, ys, zs)
    ax.view_init(50, -20 - 90)
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    target = np.mean(limits, axis=1)
    spans = np.ptp(limits, axis=1)
    box_aspect = np.asarray(ax.get_box_aspect(), dtype=np.float64)
    data_aspect = spans/box_aspect
    # Axes3D.get_proj uses _dist and _box_aspect to place its camera around
    # the limits center. Convert that actual projection geometry back to data
    # coordinates before deriving camlight left; do not force a new aspect.
    camera, light = matlab_camera_and_camlight_left(
        float(ax.azim + 90.0), float(ax.elev), target, data_aspect,
        radius=float(ax._dist))
    lit_rgba = matlab_flat_face_lighting(
        vertex_rgba, xs, ys, zs, camera, light, data_aspect,
        ambient=0.3, diffuse=0.6, specular=0.9,
        specular_exponent=10.0, backface="reverselit")
    artist = Poly3DCollection(
        quads.reshape((-1, 4, 3)),
        facecolors=np.asarray(lit_rgba).reshape((-1, 4)),
        edgecolors="none", linewidths=0, antialiased=False)
    ax.add_collection3d(artist, autolim=False)
    artist._chebfun_camlight_position = np.asarray(light)
    artist._chebfun_material = {
        "FaceLighting": "flat", "AmbientStrength": 0.3,
        "DiffuseStrength": 0.6, "SpecularStrength": 0.9,
        "SpecularExponent": 10, "BackFaceLighting": "reverselit"}
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Random2D_04.png"))
    plt.close(fig)

    # periodic random function
    ft = randnfun2(lam, DOM, seed=1, trig=True)
    _zebra(ft, 5)

    # lambda = 0.1 and 0.05
    _zebra(randnfun2(0.1, DOM, seed=2), 6)
    _zebra(randnfun2(0.05, DOM, seed=3), 7)


if __name__ == "__main__":
    run()
