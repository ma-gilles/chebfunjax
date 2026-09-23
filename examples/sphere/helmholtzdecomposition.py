"""Helmholtz-Hodge decomposition of a vector field.

Translation of sphere/HelmholtzDecomposition.m by Alex Townsend
and Grady Wright (May 2016): any tangent vector field on the sphere
splits uniquely as f = grad(phi) + curl(psi), where phi and psi solve
Poisson equations with div(f) and vorticity(f) as right-hand sides.

Original: https://www.chebfun.org/examples/sphere/HelmholtzDecomposition.html
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style, quiver_sphere
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]
SIZE = (612, 279)
VIEW = dict(elev=8, azim=-36 - 90)      # MATLAB view([-36 8])


def _xyz(fn):
    """Wrap f(x, y, z) as a function of (lam, theta) on the sphere."""
    return lambda lam, th: fn(jnp.cos(lam) * jnp.sin(th),
                              jnp.sin(lam) * jnp.sin(th), jnp.cos(th))


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"HelmholtzDecomposition_{FIG[0]:02d}.png"), size=SIZE)
    plt.close(fig)


def _contour(g, ax, color, levels=10, n=200):
    """hold on, contour(g, [color '-'], 'linewidth', 2): the level curves
    of g drawn on the visible hemisphere of the current view."""
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    C = np.asarray(g(jnp.asarray(L.ravel()), jnp.asarray(T.ravel())))
    tmp = plt.figure()
    cs = tmp.add_subplot().contour(lam, th, C.reshape(L.shape), levels)
    plt.close(tmp)
    el, az = np.deg2rad(VIEW["elev"]), np.deg2rad(VIEW["azim"])
    eye = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                    np.sin(el)])
    for segs in cs.allsegs:
        for seg in segs:
            P = np.stack([np.cos(seg[:, 0]) * np.sin(seg[:, 1]),
                          np.sin(seg[:, 0]) * np.sin(seg[:, 1]),
                          np.cos(seg[:, 1])], axis=1) * 1.01
            P[P @ eye < 0] = np.nan         # hidden behind the sphere
            ax.plot(P[:, 0], P[:, 1], P[:, 2], color=color, lw=2,
                    zorder=10)


def _quiver3(F, ax=None, title="", contours=()):
    """quiver3(F), [hold on, contour(g, fmt, 'linewidth', 2)], view."""
    fig, ax = quiver_sphere(F, ax=ax)
    for g, color in contours:
        _contour(g, ax, color)
    ax.view_init(**VIEW)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=8, pad=0)
    if title:
        ax.set_title(title, fontsize=10, pad=0)
    return fig, ax


def _single(fig, ax):
    """Leave room above a full-canvas 3-D axes for its title."""
    ax.set_position([0.0, 0.0, 1.0, 0.9])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    return fig


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    f = Spherefunv.from_functions(
        _xyz(lambda x, y, z: y * z * jnp.cos(x * y * z)),
        _xyz(lambda x, y, z: x * z * jnp.sin(4 * x + .1 * y + 5 * z**2)),
        _xyz(lambda x, y, z: 1 + x * y * z))
    _save(_single(*_quiver3(f)))

    # Project onto the tangent space.
    f = f.tangent()
    _save(_single(*_quiver3(f)))

    # 2. The curl-free component.
    phi = Spherefun.poisson(f.divergence(), 0, 251)
    _save(_single(*_quiver3(phi.gradient(),
                            title="Curl-free component of f",
                            contours=[(phi, "b")])))
    print("ans =")
    print(f"     {float(phi.gradient().vorticity().norm()):.15e}")

    # 3. The divergence-free component.
    psi = Spherefun.poisson(f.vorticity(), 0, 251)
    _save(_single(*_quiver3(psi.curl(),
                            title="Divergence-free component of f",
                            contours=[(psi, "r")])))
    print("ans =")
    print(f"     {float(psi.curl().divergence().norm()):.15e}")

    # 4. Plotting the decomposition.
    fig = plt.figure(figsize=(6.12, 2.79))
    for k, (F, ttl) in enumerate([(phi.gradient(), "Curl-free"),
                                  (psi.curl(), "Divergence-free"),
                                  (f, "Tangent vector field")]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        _quiver3(F, ax=ax, title=ttl)
    _save(fig)

    h = phi.gradient() + psi.curl()
    print("ans =")
    print(f"     {float((f - h).norm()):.15e}")

    # 5. The helmholtzdecomp command.
    phi, psi = f.helmholtzdecomp()
    _save(_single(*_quiver3(
        f, title=r"f (arrows), $\phi$ (blue), and $\psi$ (red)",
        contours=[(phi, "b"), (psi, "r")])))


if __name__ == "__main__":
    run()
