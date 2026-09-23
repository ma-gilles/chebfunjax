"""The Rayleigh quotient on the sphere.

Translation of sphere/RayleighQuotientExample.m by Grady Wright
(February 2017): the eigenvalues of a random symmetric 3x3 matrix A
recovered by maximizing the Rayleigh quotient q = x'Ax over the sphere
-- lambda1 from max2(q), lambda2 from the max of q restricted (as a
trig chebfun) to the great circle orthogonal to the first eigenvector,
and lambda3 a quarter turn further along that circle.

MATLAB's ``rng(52509); rand(3)`` is the MT19937 stream of
``numpy.random.RandomState(52509)`` filled column-major, so A is the
published matrix.

Original: https://www.chebfun.org/examples/sphere/RayleighQuotientExample.html
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import PARULA, chebfun_style, matlab_plot, plot_sphere
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.spherefun.spherefun import Spherefun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]
MS, LW = 22, 2
EL, AZ = 30, -37.5 - 90                   # MATLAB view(3)
EYE = np.array([np.cos(np.deg2rad(EL)) * np.cos(np.deg2rad(AZ)),
                np.cos(np.deg2rad(EL)) * np.sin(np.deg2rad(AZ)),
                np.sin(np.deg2rad(EL))])


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"RayleighQuotientExample_{FIG[0]:02d}.png"), size=(600, 269))
    plt.close(fig)


def _axes3():
    fig = plt.figure(figsize=(6.0, 2.69))
    ax = fig.add_axes([0.1, 0.0, 0.65, 1.0], projection="3d")
    return fig, ax


def _view(ax, ticks=(-1, 0, 1)):
    ax.view_init(elev=EL, azim=AZ)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=8, pad=0)


def _front(P):
    """Hide the part of a curve on the far side of the unit sphere."""
    P = np.array(P, dtype=float)
    P[P @ EYE < -1e-3] = np.nan
    return P


def _plot_q(q, ax):
    """plot(q), hold on."""
    plot_sphere(q, ax=ax)
    _view(ax)


def _contour_segments(g, levels, n=200):
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    C = np.asarray(g(jnp.asarray(L.ravel()), jnp.asarray(T.ravel())))
    tmp = plt.figure()
    cs = tmp.add_subplot().contour(lam, th, C.reshape(L.shape), levels)
    plt.close(tmp)
    for segs in cs.allsegs:
        for seg in segs:
            yield np.stack([np.cos(seg[:, 0]) * np.sin(seg[:, 1]),
                            np.sin(seg[:, 0]) * np.sin(seg[:, 1]),
                            np.cos(seg[:, 1])], axis=1)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    rs = np.random.RandomState(52509)
    A = 10 * (2 * rs.random_sample(9).reshape(3, 3, order="F") - 1)
    A = 0.5 * (A + A.T)

    def q_cart(x, y, z):
        return (A[0, 0] * x * x + A[1, 1] * y * y + A[2, 2] * z * z
                + 2 * A[0, 1] * x * y + 2 * A[0, 2] * x * z
                + 2 * A[1, 2] * y * z)

    q = Spherefun.from_function(
        lambda lam, th: q_cart(jnp.cos(lam) * jnp.sin(th),
                               jnp.sin(lam) * jnp.sin(th), jnp.cos(th)))

    # plot(q), hold on, contour(q, 20, 'k-'), colorbar, hold off
    fig, ax = _axes3()
    _plot_q(q, ax)
    for P in _contour_segments(q, 20):
        P = _front(P * 1.005)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], "k-", lw=0.5, zorder=10)
    lam = np.linspace(-np.pi, np.pi, 200)
    L, T = np.meshgrid(lam, np.linspace(0, np.pi, 200))
    V = np.asarray(q(jnp.asarray(L.ravel()), jnp.asarray(T.ravel())))
    cax = fig.add_axes([0.86, 0.1, 0.02, 0.84])
    fig.colorbar(ScalarMappable(norm=Normalize(V.min(), V.max()),
                                cmap=PARULA), cax=cax)
    _save(fig)

    # 3. Demonstration of the maximum principle.
    lambda1, loc = q.max2()
    lambda1 = float(lambda1)
    loc = np.asarray(loc, dtype=float).ravel()
    print("lambda1 =")
    print(f"   {lambda1:.15f}")
    lamA = np.sort(np.linalg.eigvalsh(A))[::-1]
    print("error =")
    print(f"     {abs(lamA[0] - lambda1):.15e}")

    def s2c(u):
        return np.array([np.cos(u[0]) * np.sin(u[1]),
                         np.sin(u[0]) * np.sin(u[1]), np.cos(u[1])])

    x1 = s2c(loc)
    # q is even, so x1 and -x1 both attain lambda1.  MATLAB's max2
    # returned the maximizer on the side facing the view(3) camera
    # (the red dot of the published figures); use that representative.
    if x1 @ EYE < 0:
        loc = np.array([np.angle(-np.exp(1j * loc[0])), np.pi - loc[1]])
        x1 = s2c(loc)

    def dot(ax, x):
        ax.plot([x[0]], [x[1]], [x[2]], "r.", ms=MS / 4, zorder=11)

    fig, ax = _axes3()
    _plot_q(q, ax)
    dot(ax, x1)
    _save(fig)

    # The great circle normal to x1.
    def xp(t):
        return (np.cos(loc[0]) * np.cos(loc[1]) * np.cos(t)
                - np.sin(loc[0]) * np.sin(t))

    def yp(t):
        return (np.sin(loc[0]) * np.cos(loc[1]) * np.cos(t)
                + np.cos(loc[0]) * np.sin(t))

    def zp(t):
        return -np.sin(loc[1]) * np.cos(t)

    t = np.linspace(-np.pi, np.pi, 501)
    fig, ax = _axes3()
    _plot_q(q, ax)
    P = _front(np.stack([xp(t), yp(t), zp(t)], axis=1) * 1.005)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "r-", lw=LW / 2, zorder=10)
    dot(ax, x1)
    _save(fig)

    # f = q on the great circle.
    f = chebfun(lambda t: q_cart(xp(t), yp(t), zp(t)),
                domain=(-np.pi, np.pi), trig=True)
    fig, ax = plt.subplots(figsize=(6.0, 2.69))
    matlab_plot(f, ax=ax, linewidth=LW)
    _save(fig)

    locf, lambda2 = f.max()            # (x_max, f_max)
    lambda2, locf = float(lambda2), float(locf)
    x2 = np.array([xp(locf), yp(locf), zp(locf)])
    print("lambda2 =")
    print(f"   {lambda2:.15f}")
    print("error =")
    print(f"     {abs(lamA[1] - lambda2):.15e}")

    lambda3 = float(f(locf + np.pi / 2))
    x3 = np.array([xp(locf + np.pi / 2), yp(locf + np.pi / 2),
                   zp(locf + np.pi / 2)])
    print("lambda3 =")
    print(f"  {lambda3:.15f}")
    print("error =")
    print(f"     {abs(lamA[2] - lambda3):.15e}")

    # The eigenvectors, drawn from the origin, over plot(q), alpha(0.8).
    X = np.full((9, 3), np.nan)
    X[[0, 3, 6]] = 0.0
    X[[1, 4, 7]] = np.stack([x1, x2, x3])

    def eigvecs(ax):
        ax.plot(X[:, 0], X[:, 1], X[:, 2], "r.-", lw=LW / 4,
                ms=MS / 4, zorder=11)

    fig, ax = _axes3()
    _plot_q(q, ax)
    for c in ax.collections:
        c.set_alpha(0.8)
    ax.computed_zorder = False
    eigvecs(ax)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    _save(fig)

    # 4. Zero-level curves of the surface gradient of q.
    Gq = q.gradient().components
    fig, ax = _axes3()
    u = np.linspace(-np.pi, np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    ax.plot_surface(0.99 * np.outer(np.cos(u), np.sin(v)),
                    0.99 * np.outer(np.sin(u), np.sin(v)),
                    0.99 * np.outer(np.ones_like(u), np.cos(v)),
                    color=(0.98, 0.98, 0.98), linewidth=0, shade=False)
    ax.computed_zorder = False
    for g, col in zip(Gq, ["k", "b", "m"]):
        for P in _contour_segments(g, [0.0]):
            P = _front(P)
            ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", color=col, lw=0.6)
    # The opaque sphere hides the segments to the origin; only the
    # eigenvector tips facing the viewer show.
    tips = _front(X[[1, 4, 7]])
    ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], "r.", ms=MS / 4)
    _view(ax, ticks=(-0.5, 0, 0.5))
    ax.set_box_aspect((1, 1, 1))
    _save(fig)


if __name__ == "__main__":
    run()
