"""Helmholtz decomposition of a vector field in the ball.

Faithful replica of sphere/HelmholtzDecompositionBall.m: a general
field in the unit ball splits as

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

from chebfunjax.ballfun._pt import helmholtz_decomposition
from chebfunjax.ballfun.ballfunv import Ballfunv
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')


def _quiver(ax, v, title, nr=3, nl=14, nt=7):
    rs = np.linspace(0.35, 0.95, nr)
    lam = np.linspace(-np.pi, np.pi, nl, endpoint=False)
    th = np.linspace(0.35, np.pi - 0.35, nt)
    R, L, T = np.meshgrid(rs, lam, th, indexing="ij")
    sh = R.shape
    comp = [np.asarray(c(R.reshape(1, -1), L.reshape(1, -1),
                         T.reshape(1, -1))).reshape(sh)
            for c in v.components]
    X = R * np.cos(L) * np.sin(T)
    Y = R * np.sin(L) * np.sin(T)
    Z = R * np.cos(T)
    ax.quiver(X, Y, Z, comp[0], comp[1], comp[2], length=0.2, lw=0.6,
              color="tab:blue")
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    v = Ballfunv.from_functions(
        lambda x, y, z: np.cos(x * y) * z,
        lambda x, y, z: np.sin(x * z),
        lambda x, y, z: y * z)

    f, Ppsi, Tpsi, phi = helmholtz_decomposition(v, nargout=4)
    gf = Ballfunv(*f.grad())
    print("ans =")
    print(f"     {float(gf.curl().norm()):.15e}")

    gphi = Ballfunv(*phi.grad())
    lap = Ballfunv(gphi.components[0].laplacian(),
                   gphi.components[1].laplacian(),
                   gphi.components[2].laplacian())
    print("ans =")
    print(f"     {float(lap.norm()):.15e}")

    psi = Ballfunv.PT2ballfunv(Ppsi, Tpsi)
    cp = psi.curl()
    print("ans =")
    print(f"     {float(cp.div().norm()):.15e}")

    fig = plt.figure(figsize=(10.8, 10.0))
    for i, (vv, ttl) in enumerate([
            (v, "vector field"), (gf, "curl-free"),
            (cp, "divergence-free"), (gphi, "harmonic")]):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        _quiver(ax, vv, ttl)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, "HelmholtzDecompositionBall_repl_01.png"),
        dpi=140, bbox_inches="tight")
    plt.close(fig)

    w = gf + cp + gphi
    print("ans =")
    print(f"     {float((v - w).norm()):.15e}")


if __name__ == "__main__":
    run()
