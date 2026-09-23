"""The Laplace equation on the unit ball.

Translation of sphere/LaplaceBall.m by Nick Trefethen (June
2019): solve lap(u) = 0 in the ball with smooth random boundary data
h (characteristic wavelength lambda = 0.2), via the ballfun Helmholtz
solver with K = 0 and Dirichlet data.  The checks are the published
self-consistency identities: u matches h on the boundary (at (1,0,0)
and at Oxford's coordinates), u(0) equals the mean of h, and the mean
over the inner sphere r = 1/2 equals the same value.

The random sample uses a seeded harmonic expansion (MATLAB's rng(1)
stream is not reproducible); every printed identity is
sample-independent.

Original: https://www.chebfun.org/examples/sphere/LaplaceBall.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.plotting import PARULA, chebfun_style, plot_ball_slices
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.spherefun.spherefun import _real_ylm_values

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]

LAM = 0.2
DEG = int(np.floor(2 * np.pi / LAM))    # max harmonic degree ~ 31
RNG = np.random.default_rng(1)

# Random harmonic coefficients, unit pointwise variance
# (sum of (2l+1)|Y|^2 over a shell = (2l+1)/(4pi)).
_COEF = {}
_nm = (DEG + 1) ** 2
for _l in range(DEG + 1):
    for _m in range(-_l, _l + 1):
        _COEF[(_l, _m)] = RNG.standard_normal() * np.sqrt(
            4 * np.pi / _nm)


def h_eval(lam, th, r_scale=None):
    out = np.zeros(np.shape(np.asarray(lam)))
    for (l, m), c in _COEF.items():
        cc = c if r_scale is None else c * r_scale**l
        out = out + cc * np.asarray(_real_ylm_values(l, m, lam, th))
    return out


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, f"LaplaceBall_{FIG[0]:02d}.png"),
             size=(600, 253))
    plt.close(fig)


def _plot_fn(F, clim=None, n=200, axis_off=False):
    """plot(F) of a function on the sphere, with a colorbar."""
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    V = np.asarray(F(L.ravel(), T.ravel())).reshape(L.shape)
    fig = plt.figure(figsize=(6.0, 2.53))
    ax = fig.add_axes([0.1, 0.0, 0.65, 1.0], projection="3d")
    X, Y, Z = (np.cos(L) * np.sin(T), np.sin(L) * np.sin(T), np.cos(T))
    norm = Normalize(*(clim if clim else (V.min(), V.max())))
    ax.plot_surface(X, Y, Z, facecolors=PARULA(norm(V)), rstride=1,
                    cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.view_init(elev=30, azim=-127.5)      # MATLAB view(3)
    if axis_off:
        ax.set_axis_off()
    cax = fig.add_axes([0.84, 0.12, 0.025, 0.8])
    fig.colorbar(ScalarMappable(norm=norm, cmap=PARULA), cax=cax)
    _save(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    _plot_fn(h_eval, clim=(-2, 2), axis_off=True)

    # Evaluations of the boundary data (cartesian and spherical forms).
    print("ans =")
    print(f"  {float(h_eval(0.0, np.pi / 2)):.15f}")
    print("ans =")
    print(f"  {float(h_eval(0.0, np.pi / 2)):.15f}")
    meanh = _COEF[(0, 0)] / np.sqrt(4 * np.pi)
    print("meanh =")
    print(f"  {meanh:.15f}")

    # Solve the Laplace problem with the Helmholtz solver (K = 0).
    m = 2 * DEG + 6
    u = Ballfun.helmholtz(lambda x, y, z: 0.0 * x, 0.0,
                          lambda lam, th: h_eval(lam, th),
                          m, n=2 * m, p=2 * m + 1)
    print("ans =")                         # h(1,0,0)
    print(f"  {float(h_eval(0.0, np.pi / 2)):.15f}")
    print("ans =")                         # u(1,0,0)
    print(f"  {float(u(1.0, 0.0, np.pi / 2)):.15f}")

    # Oxford's coordinates.
    long = -1.26 * np.pi / 180
    lat = 51.75 * np.pi / 180
    print("ans =")
    print(f"  {float(h_eval(long, np.pi / 2 - lat)):.15f}")
    print("ans =")
    print(f"  {float(u(1.0, long, np.pi / 2 - lat)):.15f}")

    # The value at the origin equals the mean of the boundary data.
    print("meanh =")
    print(f"  {meanh:.15f}")
    print("ans =")
    print(f"  {float(u(1e-14, 0.0, np.pi / 2)):.15f}")

    # plot(u): the solution looks uniform in the interior.
    fig = plt.figure(figsize=(6.0, 2.53))
    ax = fig.add_subplot(111, projection="3d")
    plot_ball_slices(u, ax=ax, azim=-127.5)
    _save(fig)

    # The solution on the inner sphere r = 1/2.
    _plot_fn(lambda lam, th: np.asarray(
        u(0.5 * np.ones((1, np.size(lam))),
          np.reshape(lam, (1, -1)),
          np.reshape(th, (1, -1)))).ravel())
    # Exact interior harmonic extension for comparison.
    exact_inner = lambda lam, th: h_eval(lam, th, r_scale=0.5)  # noqa: E731
    n = 200
    lamg = np.linspace(-np.pi, np.pi, n)
    thg = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lamg, thg)
    Ui = np.asarray(u(0.5 * np.ones_like(L), L, T))
    Ue = exact_inner(L.ravel(), T.ravel()).reshape(L.shape)
    assert np.max(np.abs(Ui - Ue)) < 1e-10   # interior harmonic extension
    # Mean over the inner sphere (Gauss-Legendre in cos theta).
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(80)
    thq = np.arccos(xg)
    lamq = -np.pi + 2 * np.pi * np.arange(160) / 160
    LQ, TQ = np.meshgrid(lamq, thq)
    UQ = np.asarray(u(0.5 * np.ones_like(LQ), LQ, TQ))
    mean_inner = float(np.sum(UQ * wg[:, None]) * (2 * np.pi / 160)
                       / (4 * np.pi))
    print("meanh =")
    print(f"  {meanh:.15f}")
    print("ans =")                         # mean2(uinner)
    print(f"  {mean_inner:.15f}")


if __name__ == "__main__":
    run()
