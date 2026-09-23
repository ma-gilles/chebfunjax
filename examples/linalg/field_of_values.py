"""Field of values.

Translation of linalg/FieldOfValues.m by Nick Trefethen
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
import inspect
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax.chebfun1d.fov as fov_module
from chebfunjax.chebfun1d.fov import fov
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

FIG = [0]


def _num(v):
    """MATLAB format-long display of a real scalar."""
    return f"{int(v):6d}" if float(v) == int(v) else f"   {v:.15f}"


def _save(fig, close=True):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(
        _IMG, f"FieldOfValues_{FIG[0]:02d}.png"))
    if close:
        plt.close(fig)


def _plot_fov(F, eigs, axis_lim=None):
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    bps = [float(v) for v in F.domain.breakpoints]
    for a, b in zip(bps[:-1], bps[1:]):
        t = np.linspace(a, b, 300)
        v = np.asarray(F(t))
        ax.plot(v.real, v.imag, 'b', lw=1.6)
    # 'jumpline' {'b'}: join the one-sided values at each interior break
    for bp in bps[1:-1]:
        zl = complex(np.asarray(F(bp, 'left')))
        zr = complex(np.asarray(F(bp, 'right')))
        ax.plot([zl.real, zr.real], [zl.imag, zr.imag], 'b', lw=1.6)
    ax.plot(eigs.real, eigs.imag, '.k', ms=12)
    ax.set_aspect("equal")
    ax.grid(True)
    if axis_lim:
        ax.axis(axis_lim)
    else:
        ax.margins(0.1)  # axis(1.1*ax)
    return fig, ax


def run():
    os.makedirs(_IMG, exist_ok=True)

    rs = np.random.RandomState(1)
    A = rs.randn(20, 20)
    FA = fov(A)
    eigsA = np.linalg.eigvals(A)
    fig, ax = _plot_fov(FA, eigsA)
    _save(fig, close=False)

    reF = FA.real()
    maxtheta, alpha = reF.max()
    print("alpha =")
    print(f"   {float(alpha):.15f}")
    print("maxtheta =")
    print(_num(float(maxtheta)))
    zmax = complex(np.asarray(FA(float(maxtheta))))
    ax.plot(zmax.real, zmax.imag, '.r', ms=18)
    _save(fig)

    alpha2 = np.max(np.linalg.eigvalsh((A + A.T) / 2))
    print("alpha =")
    print(f"   {alpha2:.15f}")

    B = np.diag(eigsA)
    FB = fov(B)
    fig, ax = _plot_fov(FB, eigsA)
    reB = FB.real()
    mth, _ = reB.max()
    zb = complex(np.asarray(FB(float(mth))))
    ax.plot(zb.real, zb.imag, '.r', ms=18)
    _save(fig)
    print("FB =")
    print(repr(FB))

    C = np.array([[0, 3, 0, 0],
                  [-3, 0, 0, 0],
                  [0, 0, 0, 3],
                  [0, 0, 1, 1]], dtype=float)
    print("C =")
    for row in C:
        print("".join(f"{int(v):6d}" for v in row))
    FC = fov(C)
    eigsC = np.linalg.eigvals(C)
    fig, ax = _plot_fov(FC, eigsC, axis_lim=[-4, 4, -4, 4])
    _save(fig)

    print(inspect.getsource(fov_module))


if __name__ == "__main__":
    run()
