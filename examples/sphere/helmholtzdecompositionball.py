"""Helmholtz decomposition of a vector field in the ball.

Translation of sphere/HelmholtzDecompositionBall.m by Nicolas Boulle
and Alex Townsend (May 2019): a general field in the unit ball splits as

    v = grad(f) + curl(psi) + grad(phi)

with f from a Poisson solve on div(v), phi harmonic matching the
normal boundary flux (Laplace-Neumann solve), and psi in
poloidal-toroidal form from the remaining divergence-free part.

Original: https://www.chebfun.org/examples/sphere/HelmholtzDecompositionBall.html
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

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.ballfun.ballfunv import Ballfunv
from chebfunjax.plotting import chebfun_style, quiver_ball
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"HelmholtzDecompositionBall_{FIG[0]:02d}.png"),
        size=(600, 253.4))  # 2.53 in * 100 dpi floors to 252 px
    plt.close(fig)


def _quiver(v, title="", ax=None, numpts=25):
    """quiver(v, 'numpts', numpts), title(title)."""
    single = ax is None
    fig, ax = quiver_ball(v, ax=ax, n_pts=numpts, title=title)
    for set_ticks in (ax.set_xticks, ax.set_yticks, ax.set_zticks):
        set_ticks([-1, 0, 1])
    ax.tick_params(labelsize=7, pad=-2)
    if single:
        ax.set_position([0.0, 0.0, 1.0, 0.9])
    return fig


def _grad(f):
    return Ballfunv(*f.grad())


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    v = Ballfunv.from_functions(
        lambda x, y, z: np.cos(x * y) * z,
        lambda x, y, z: np.sin(x * z),
        lambda x, y, z: y * z)
    _save(_quiver(v))

    # Curl-free component: Delta f = div(v), f = 0 on the boundary.
    f = Ballfun.helmholtz(v.div(), 0.0, lambda lam, th: 0.0 * lam,
                          50, 50, 50)
    gf = _grad(f)
    _save(_quiver(gf, "curl-free component of v"))
    print("ans =")
    print(f"     {float(gf.curl().norm()):.15e}")

    # Harmonic component: Delta phi = 0, dphi/dr = n . v1 on r = 1.
    v1 = v - gf

    def bc(lam, th):
        lam, th = np.asarray(lam), np.asarray(th)
        L, T = lam.reshape(1, -1), th.reshape(1, -1)
        vx, vy, vz = (np.asarray(c) for c in v1(np.ones_like(L), L, T))
        vn = (vx * np.cos(L) * np.sin(T) + vy * np.sin(L) * np.sin(T)
              + vz * np.cos(T))
        return np.real(vn).reshape(lam.shape)

    zero = Ballfun.from_function(lambda x, y, z: 0.0 * x)
    phi = Ballfun.helmholtz(zero, 0.0, bc, 50, 50, 50, bc_type="neumann")
    gphi = _grad(phi)
    _save(_quiver(gphi, "harmonic component of v"))
    print("ans =")
    print(f"     {float(gphi.laplacian().norm()):.15e}")

    # Divergence-free component via the poloidal-toroidal decomposition.
    v2 = v1 - gphi
    Pv, Tv = v2.PTdecomposition()
    Ppsi = Ballfun.helmholtz(-Tv, 0.0, lambda lam, th: 0.0 * lam,
                             50, 50, 50)
    Tpsi = Pv
    psi = Ballfunv.PT2ballfunv(Ppsi, Tpsi)
    cp = psi.curl()
    _save(_quiver(cp, "divergence-free component of v"))
    print("ans =")
    print(f"     {float(cp.div().norm()):.15e}")

    # Visualizing the decomposition.
    fig = plt.figure(figsize=(6.0, 2.53))
    for k, (vv, ttl) in enumerate([(v, "vector field"), (gf, "curl-free"),
                                   (cp, "divergence-free"),
                                   (gphi, "harmonic")]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        _quiver(vv, ttl, ax=ax, numpts=20)
    _save(fig)

    w = gf + cp + gphi
    print("ans =")
    print(f"     {float((v - w).norm()):.15e}")

    # The HelmholtzDecomposition command.
    f, Ppsi, Tpsi, phi = v.HelmholtzDecomposition(nargout=4)
    psi = Ballfunv.PT2ballfunv(Ppsi, Tpsi)


if __name__ == "__main__":
    run()
