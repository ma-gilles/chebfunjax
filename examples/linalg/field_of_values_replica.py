"""Field of values.

Faithful replica of linalg/FieldOfValues.m by Nick Trefethen
(November 2010): the field of values (numerical range) of a matrix as
a chebfun of the boundary parametrized by angle, computed by
Johnson's algorithm; the numerical abscissa; and the polygonal /
line-segment cases of normal and non-generic matrices.

MATLAB seeds rng(1); randn is not bit-reproducible across systems,
so the random matrix differs; all internal consistency checks
replicate.

Original: https://www.chebfun.org/examples/linalg/FieldOfValues.html
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
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

FIG = [0]


def _fov_point(A, theta_arr):
    theta_arr = np.atleast_1d(np.asarray(theta_arr, dtype=float))
    out = np.empty(theta_arr.shape, dtype=complex)
    for i, th in enumerate(theta_arr.ravel()):
        r = np.exp(1j * th)
        H = (r * A + np.conj(r) * A.conj().T) / 2
        w, V = np.linalg.eigh(H)
        v = V[:, -1]
        out.ravel()[i] = (v.conj() @ A @ v) / (v.conj() @ v)
    return out.reshape(theta_arr.shape)


def _fov_chebfun(A, splitting=False):
    op = lambda t: jnp.asarray(_fov_point(A, np.asarray(t)))  # noqa: E731
    f = cj.chebfun(op, domain=(0.0, 2 * np.pi),
                   splitting=splitting)
    return f.merge() if splitting else f


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"FieldOfValues_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_fov(F, eigs, axis_lim=None):
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    bps = [float(v) for v in F.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 300)
        v = np.asarray(F(t))
        ax.plot(v.real, v.imag, 'b', lw=1.6)
    ax.plot(eigs.real, eigs.imag, '.k', ms=12)
    ax.set_aspect("equal")
    ax.grid(True)
    if axis_lim:
        ax.axis(axis_lim)
    return fig, ax


def run():
    os.makedirs(_IMG, exist_ok=True)

    rs = np.random.RandomState(1)
    A = rs.randn(20, 20)
    FA = _fov_chebfun(A)
    eigsA = np.linalg.eigvals(A)
    fig, ax = _plot_fov(FA, eigsA)

    reF = FA.real()
    maxtheta, alpha = reF.max()
    print("alpha =")
    print(f"   {float(alpha):.15f}")
    print("maxtheta =")
    print(f"     {float(maxtheta):.6g}")
    zmax = complex(np.asarray(FA(float(maxtheta))))
    ax.plot(zmax.real, zmax.imag, '.r', ms=18)
    _save(fig)

    alpha2 = np.max(np.linalg.eigvalsh((A + A.T) / 2))
    print("alpha =")
    print(f"   {alpha2:.15f}")

    B = np.diag(eigsA)
    FB = _fov_chebfun(B, splitting=True)
    fig, ax = _plot_fov(FB, eigsA)
    reB = FB.real()
    mth, _ = reB.max()
    zb = complex(np.asarray(FB(float(mth))))
    ax.plot(zb.real, zb.imag, '.r', ms=18)
    _save(fig)
    print("FB =")
    print(repr(FB)[:800])

    C = np.array([[0, 3, 0, 0],
                  [-3, 0, 0, 0],
                  [0, 0, 0, 3],
                  [0, 0, 1, 1]], dtype=float)
    print("C =")
    for row in C:
        print("  " + "".join(f"{int(v):6d}" for v in row))
    FC = _fov_chebfun(C, splitting=True)
    eigsC = np.linalg.eigvals(C)
    fig, ax = _plot_fov(FC, eigsC, axis_lim=[-4, 4, -4, 4])
    _save(fig)


if __name__ == "__main__":
    run()
