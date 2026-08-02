"""Nearest orthonormal functions.

Faithful replica of approx/NearestOrthFun.m by Behnam Hashemi (July
2016): the nearest quasimatrix with orthonormal columns, Q = U V^T
from the quasimatrix SVD, compared with QR orthonormalization.

Original: https://www.chebfun.org/examples/approx/NearestOrthFun.html
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
from chebfunjax.chebfun1d.linalg import Quasimatrix, qr_quasimatrix, svd_quasimatrix
from chebfunjax.plotting import chebfun_style
from chebfunjax.utils.gallery import gallery

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

FIGNUM = [0]


def _gram(cols):
    m = len(cols)
    G = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            G[i, j] = float((cols[i] * cols[j]).sum())
    return G


def _dist_fro(colsA, colsB):
    return np.sqrt(sum(float(((ca - cb) ** 2).sum())
                       for ca, cb in zip(colsA, colsB)))


def nearest_ortho(cols, label):
    from chebfunjax.domain import Domain
    dom = Domain((float(cols[0].domain.a), float(cols[0].domain.b)))
    A = Quasimatrix(cols=list(cols), domain=dom)
    U, S, V = svd_quasimatrix(A)
    V = np.asarray(V)
    # Q = U * V^T: columns q_j = sum_k u_k V[j,k]
    Q = [sum(float(V[j, k]) * U.cols[k] for k in range(len(cols)))
         for j in range(len(cols))]
    Q2m, _R = qr_quasimatrix(A)
    Q2 = list(Q2m.cols)
    m = len(cols)

    FIGNUM[0] += 1
    a = float(cols[0].domain.a)
    b = float(cols[0].domain.b)
    xs = np.linspace(a, b, 1500)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    vmax = 0.0
    for q in Q2:
        v = np.asarray(q(jnp.asarray(xs)))
        vmax = max(vmax, np.max(np.abs(v)))
        axes[0].plot(xs, v, lw=1.1)
    for q in Q:
        axes[1].plot(xs, np.asarray(q(jnp.asarray(xs))), lw=1.1)
    for ax, t in zip(axes, ("QR orthonormalization",
                            "Optimal orthonormalization")):
        ax.set_ylim(-1.05 * vmax, 1.05 * vmax)
        ax.grid(True)
        ax.set_title(t, fontsize=11)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"NearestOrthFun_repl_{FIGNUM[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Departure from orthogonality in the columns of A = "
          f"{np.linalg.norm(_gram(cols) - np.eye(m), 2):3.2f}")
    print(f"Departure from orthogonality in the columns of Q = "
          f"{np.linalg.norm(_gram(Q) - np.eye(m), 2):3.2g}")
    print(f"Departure from orthogonality in the columns of Q2 = "
          f"{np.linalg.norm(_gram(Q2) - np.eye(m), 2):3.2g}")
    print(f"The distance between A and Q2 = {_dist_fro(cols, Q2):3.2f}")
    print(f"The distance between A and its closest orthonormal "
          f"quasimatrix = {_dist_fro(cols, Q):3.2f}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    x = cj.chebfun(lambda t: t)

    # Vandermonde quasimatrix of monomials
    cols = [x**0, x, x**2, x**3, x**4, x**5]
    nearest_ortho(cols, "vandermonde")

    # Chebyshev-Vandermonde restricted to [0, 1]
    colsT = [cj.chebfun(lambda t, k=k: jnp.cos(
        k * jnp.arccos(jnp.clip(t, -1.0, 1.0))), domain=(0.0, 1.0))
        for k in range(6)]
    nearest_ortho(colsT, "vandercheb-restricted")

    # A mixed smooth quasimatrix
    cols = [x**0, x.cos(), (x**2).sin(), x**3, x**4, x**5]
    nearest_ortho(cols, "mixed")

    # Gallery functions: stegosaurus, wiggly, blasius (different domains
    # -> map all to a common domain by construction on their own domain
    # is required; MATLAB concatenates on shared domains only, so we
    # rebuild each on [-1,1] via its handle where needed)
    g1 = gallery("stegosaurus")
    g2 = gallery("wiggly")
    g3 = gallery("blasius")
    # bring to common domain [0,10] (stegosaurus/blasius live there);
    # wiggly lives on [-1,1] -> rescale
    def _to_010(f):
        a, b = float(f.domain.a), float(f.domain.b)
        return cj.chebfun(lambda t: f((b - a) * t / 10.0 + a),
                          domain=(0.0, 10.0))
    cols = [_to_010(g1), _to_010(g2), _to_010(g3)]
    nearest_ortho(cols, "gallery")


if __name__ == "__main__":
    run()
