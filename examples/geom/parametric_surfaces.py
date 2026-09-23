"""Parametric surfaces.

Translation of geom/ParametricSurfaces.m by Rodrigo Platte
(March 2013): a gallery of surfaces built from chebfun2 coordinate
functions -- cones, spheres, a seashell, a Mobius strip with its
normal and tangent frames, and two Klein bottles.

Original: https://www.chebfun.org/examples/geom/ParametricSurfaces.html
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
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.plotting import PARULA, chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'geom')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"ParametricSurfaces_{FIG[0]:02d}.png"))


def _view(ax, az=-37.5, el=30):
    """MATLAB view(az, el); the default is view(3)."""
    ax.view_init(elev=el, azim=az - 90)


def _grid(r, n):
    """Values of the components of the Chebfun2v r on an n x n grid of
    its parameter domain (MATLAB meshgrid orientation)."""
    a, b, c, d = map(float, r.components[0].domain)
    U, V = np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, n))
    Uj, Vj = jnp.asarray(U), jnp.asarray(V)
    return [np.asarray(F(Uj, Vj)) for F in r.components], U, V


def _surf(r, f=None, n=150, alpha=1.0, cmap=PARULA, light=False,
          equal=True, axis_off=False, view=(-37.5, 30)):
    """surf(x, y, z[, f]) of r = [x; y; z]: colored by z (or f)."""
    (X, Y, Z), U, V = _grid(r, n)
    C = Z if f is None else np.asarray(f(jnp.asarray(U), jnp.asarray(V)))
    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    fc = cmap(Normalize(C.min(), C.max())(C))
    if light:
        # camlight: a light at the camera, Lambertian (two-sided) shading
        # from the surface normals.
        az, el = np.deg2rad(view[0] - 90), np.deg2rad(view[1])
        eye = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                        np.sin(el)])
        P = np.stack([X, Y, Z], axis=-1)
        nrm = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
        nn = np.linalg.norm(nrm, axis=-1)
        shade = np.where(nn > 0, np.abs(nrm @ eye) / np.maximum(nn, 1e-300),
                         1.0)
        fc[..., :3] *= (0.35 + 0.65 * shade)[..., None]
    fc[..., 3] = alpha
    ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    if equal:
        ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    if axis_off:
        ax.set_axis_off()
    _view(ax, *view)
    return fig, ax


def _quiver3(ax, r, F, scale, color, numpts=20):
    """hold on, quiver3(x, y, z, F, scale, color, 'numpts', numpts)."""
    (X, Y, Z), _, _ = _grid(r, numpts)
    W = _grid(F, numpts)[0] if isinstance(F, Chebfun2v) else F(numpts)
    # MATLAB autoscale: the longest arrow spans ~scale grid cells.
    h = np.mean([np.ptp(A) for A in (X, Y, Z)]) / numpts
    mag = np.sqrt(sum(w**2 for w in W)).max()
    s = scale * h / mag
    ax.quiver(X, Y, Z, *(s * w for w in W), color=color, lw=0.6,
              arrow_length_ratio=0.3)


def _vec(fx, fy, fz, domain):
    return Chebfun2v.from_functions(fx, fy, fz, domain=domain)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Cone and hyperboloid of one sheet over (u, v) in [-1,1] x [0,2pi].
    dom = (-1.0, 1.0, 0.0, 2 * np.pi)
    cone = _vec(lambda u, v: u * jnp.cos(v), lambda u, v: u * jnp.sin(v),
                lambda u, v: u + 0 * v, dom)
    fig, _ = _surf(cone)
    _save(fig)
    plt.close(fig)
    hyp = _vec(lambda u, v: jnp.sqrt(0.25 + u**2) * jnp.cos(v),
               lambda u, v: jnp.sqrt(0.25 + u**2) * jnp.sin(v),
               lambda u, v: u + 0 * v, dom)
    fig, _ = _surf(hyp)
    _save(fig)
    plt.close(fig)
    f = Chebfun2.from_function(lambda u, v: (1 - u) * jnp.sin(10 * v),
                               domain=dom)
    fig, _ = _surf(hyp, f)
    _save(fig)
    plt.close(fig)

    # Sphere, bumpy sphere, and a function on the bumpy sphere.
    dom = (0.0, 2 * np.pi, -np.pi / 2, np.pi / 2)
    sph = _vec(lambda t, p: jnp.cos(p) * jnp.cos(t),
               lambda t, p: jnp.cos(p) * jnp.sin(t),
               lambda t, p: jnp.sin(p) + 0 * t, dom)
    fig, _ = _surf(sph)
    _save(fig)
    plt.close(fig)

    def rad(t, p):
        x, y, z = jnp.cos(p) * jnp.cos(t), jnp.cos(p) * jnp.sin(t), jnp.sin(p)
        return 1 + 0.05 * (jnp.sin(20 * x) + jnp.sin(20 * y)
                           + jnp.sin(20 * z))

    bumpy = _vec(lambda t, p: rad(t, p) * jnp.cos(p) * jnp.cos(t),
                 lambda t, p: rad(t, p) * jnp.cos(p) * jnp.sin(t),
                 lambda t, p: rad(t, p) * jnp.sin(p), dom)
    fig, _ = _surf(bumpy, n=300)
    _save(fig)
    plt.close(fig)
    f = Chebfun2.from_function(
        lambda t, p: jnp.sin(10 * t) * jnp.cos(15 * p), domain=dom)
    fig, ax = _surf(bumpy, f, n=300)
    _save(fig)
    _view(ax, 0, 90)                             # top view
    _save(fig)
    plt.close(fig)

    # Seashell.
    dom = (0.0, 6 * np.pi, 0.0, 2 * np.pi)
    shell = _vec(
        lambda u, v: 2 * (1 - jnp.exp(u / (6 * np.pi))) * jnp.cos(u)
        * jnp.cos(v / 2)**2,
        lambda u, v: 2 * (-1 + jnp.exp(u / (6 * np.pi))) * jnp.sin(u)
        * jnp.cos(v / 2)**2,
        lambda u, v: 1 - jnp.exp(u / (3 * np.pi)) - jnp.sin(v)
        + jnp.exp(u / (6 * np.pi)) * jnp.sin(v), dom)
    fig, ax = _surf(shell, n=250, light=True, view=(160, 10))
    _save(fig)
    _view(ax, -180, 90)                          # top view
    _save(fig)
    plt.close(fig)

    # Moebius strip, its normal field, and its tangent fields.
    dom = (0.0, 2 * np.pi, -1.0, 1.0)
    r = _vec(lambda u, v: (1 + 0.5 * v * jnp.cos(u / 2)) * jnp.cos(u),
             lambda u, v: (1 + 0.5 * v * jnp.cos(u / 2)) * jnp.sin(u),
             lambda u, v: 0.5 * v * jnp.sin(u / 2), dom)
    fig, ax = _surf(r, light=True)
    _save(fig)
    n = r.normal()
    _quiver3(ax, r, n, 2, "k", numpts=8)
    _save(fig)
    ru = r.diff(1, 1)
    rv = r.diff(1, 2)
    _quiver3(ax, r, ru, 1, "r", numpts=8)
    _quiver3(ax, r, rv, 1, "b", numpts=8)
    _view(ax, -80, 65)
    _save(fig)
    plt.close(fig)

    ip = ru.dot(rv)
    ip = Chebfun2.from_function(lambda u, v: ip(u, v), domain=dom)
    print("ans =")
    print(f"     {float(ip.norm(np.inf)):.15e}")

    # Project V = [sin(5u); cos(5v); 0] onto the surface:
    # PV = (R1'V) R1 + (R2'V) R2 with R1, R2 the unit tangents.
    def pv(numpts):
        (a1, a2, a3), U, V = _grid(ru, numpts)
        (b1, b2, b3), _, _ = _grid(rv, numpts)
        Vf = (np.sin(5 * U), np.cos(5 * V), 0 * U)
        R1 = [a / np.sqrt(a1**2 + a2**2 + a3**2) for a in (a1, a2, a3)]
        R2 = [b / np.sqrt(b1**2 + b2**2 + b3**2) for b in (b1, b2, b3)]
        c1 = sum(p * q for p, q in zip(R1, Vf))
        c2 = sum(p * q for p, q in zip(R2, Vf))
        return [c1 * p + c2 * q for p, q in zip(R1, R2)]

    fig, ax = _surf(r, light=True, equal=False, view=(-80, 65))
    _quiver3(ax, r, pv, 2, "k", numpts=30)
    _save(fig)
    plt.close(fig)

    # Klein bagel ("figure 8" immersion), top and side views.
    dom = (0.0, 2 * np.pi, 0.0, 2 * np.pi)
    rr = 3

    def ring(u, v):
        return rr + jnp.cos(u / 2) * jnp.sin(v) - jnp.sin(u / 2) * jnp.sin(2 * v)

    bagel = _vec(lambda u, v: ring(u, v) * jnp.cos(u),
                 lambda u, v: ring(u, v) * jnp.sin(u),
                 lambda u, v: jnp.sin(u / 2) * jnp.sin(v)
                 + jnp.cos(u / 2) * jnp.sin(2 * v), dom)
    fig, ax = _surf(bagel, alpha=0.6, cmap=plt.get_cmap("hot"), light=True,
                    axis_off=True)
    _save(fig)
    _view(ax, 0, 90)
    _save(fig)
    _view(ax, 90, 0)
    _save(fig)
    plt.close(fig)

    # Robert Israel's immersion of the Klein bottle and its normal field.
    dom = (0.0, np.pi, 0.0, 2 * np.pi)
    c, s = jnp.cos, jnp.sin
    klein = _vec(
        lambda u, v: -(2 / 15) * c(u) * (
            3 * c(v) - 30 * s(u) + 90 * c(u)**4 * s(u)
            - 60 * c(u)**6 * s(u) + 5 * c(u) * c(v) * s(u)),
        lambda u, v: -(1 / 15) * s(u) * (
            3 * c(v) - 3 * c(u)**2 * c(v) - 48 * c(u)**4 * c(v)
            + 48 * c(u)**6 * c(v) - 60 * s(u) + 5 * c(u) * c(v) * s(u)
            - 5 * c(u)**3 * c(v) * s(u) - 80 * c(u)**5 * c(v) * s(u)
            + 80 * c(u)**7 * c(v) * s(u)),
        lambda u, v: (2 / 15) * (3 + 5 * c(u) * s(u)) * s(v), dom)
    fig, ax = _surf(klein, alpha=0.5, light=True, axis_off=True)
    _save(fig)
    _quiver3(ax, klein, klein.normal() * -1, 2, "k")
    _save(fig)
    plt.close(fig)


if __name__ == "__main__":
    run()
