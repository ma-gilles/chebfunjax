"""Rotating functions on the sphere.

Translation of sphere/SpherefunRotate.m by Alex Townsend and
Grady Wright (May 2017): the rotate command with ZXZ Euler angles --
integral preservation, the four-angle panel, rotation of a spherical
harmonic Y_10^3 whose coefficients stay in the degree-10 shell, and
rank growth of rotated high-rank functions.

Original: https://www.chebfun.org/examples/sphere/SpherefunRotate.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from numpy.polynomial.legendre import leggauss
from scipy.special import sph_harm_y_all

from chebfunjax.plotting import PARULA, chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.spherefun.spherefun import Spherefun, _real_ylm_values

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]


def _panel(fs_titles, fname, n=200):
    FIG[0] += 1
    k = len(fs_titles)
    ncols = min(k, 2)
    nrows = (k + 1) // 2
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    X, Y, Z = (np.cos(L) * np.sin(T), np.sin(L) * np.sin(T), np.cos(T))
    fig = plt.figure(figsize=(5.2 * ncols, 4.8 * nrows))
    for i, (F, ttl) in enumerate(fs_titles):
        V = np.asarray(F(L.ravel(), T.ravel())).reshape(L.shape)
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.plot_surface(X, Y, Z, facecolors=PARULA(
            Normalize(V.min(), V.max())(V)), rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=30, azim=-127.5)      # MATLAB view(3)
        ax.set_axis_off()
        ax.set_title(ttl, fontsize=11, pad=0)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG,
                             f"SpherefunRotate_{FIG[0]:02d}.png"),
             size=(600, 269))
    plt.close(fig)


def _randnfunsphere(lam, rng):
    """randnfunsphere(lam): random combination of the orthonormal real
    spherical harmonics up to degree floor(2*pi/lam), coefficients
    randn * sqrt(4*pi/(deg+1)^2) (MATLAB randnfunsphere.m).

    The library's randnfunsphere normalizes P_l^m with factorials and
    overflows beyond degree ~85, so the harmonics come from scipy's
    stable ``sph_harm_y_all``.  The function is sampled on a lat-lon grid
    that resolves degree ``deg`` exactly and built with
    ``Spherefun.from_values`` (MATLAB ``spherefun(F)``); adaptive
    construction of a degree-209 random function takes hours.
    """
    deg = int(np.floor(2 * np.pi / lam))
    c = rng.standard_normal((deg + 1) ** 2)
    c *= np.sqrt(4 * np.pi / c.size)
    Cc = np.zeros((deg + 1, deg + 1))      # cos(m lam) coefficient of (l, m)
    Cs = np.zeros((deg + 1, deg))          # sin(m lam), m = 1..deg
    k = 0
    for el in range(deg + 1):
        cl = c[k:k + 2 * el + 1]           # m = -l..l
        Cc[el, :el + 1] = cl[el:]
        Cs[el, :el] = cl[:el][::-1]
        k += 2 * el + 1
    M, N = 2 * deg + 4, deg + 3
    lamg = -np.pi + 2 * np.pi * np.arange(M) / M
    thg = np.linspace(0, np.pi, N)
    P = np.real(sph_harm_y_all(deg, deg, thg, 0 * thg))[:, :deg + 1, :]
    A = np.einsum("lm,lmp->mp", Cc, P)
    A[1:] *= np.sqrt(2)
    B = np.sqrt(2) * np.einsum("lm,lmp->mp", Cs, P[:, 1:, :])
    m = np.arange(deg + 1)
    F = (A.T @ np.cos(np.outer(m, lamg))
         + B.T @ np.sin(np.outer(m[1:], lamg)))
    return Spherefun.from_values(F), deg


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # f = cos(50z) + x^2 and a rotation of it.
    f = Spherefun.from_function(
        lambda lam, th: np.cos(50 * np.cos(th))
        + (np.cos(lam) * np.sin(th))**2)
    g = f.rotate(-np.pi / 4, np.pi / 2, np.pi / 8)
    _panel([(f, "Original"), (g, "Rotated")],
           "SpherefunRotate_01.png")

    h = f + g
    _panel([(h, "")], "SpherefunRotate_02.png")

    print("ans =")
    print(f"     {abs(float(f.sum2()) - float(g.sum2())):.15e}")

    # Euler angle panel for cos(50 x (y - 0.5)).
    f2 = Spherefun.from_function(
        lambda lam, th: np.cos(
            50 * (np.cos(lam) * np.sin(th))
            * (np.sin(lam) * np.sin(th) - 0.5)))
    _panel([(f2, "Original"),
            (f2.rotate(np.pi / 4, 0, 0), r"Rotated $\phi=\pi/4$"),
            (f2.rotate(0, np.pi / 4, 0), r"Rotated $\theta=\pi/4$"),
            (f2.rotate(np.pi / 4, 0, np.pi / 4),
             r"Rotated $\phi=\psi=\pi/4$")],
           "SpherefunRotate_03.png")

    # Rotating Y_10^3 keeps the coefficients in the degree-10 shell.
    Y103 = Spherefun.sphharm(10, 3)
    g3 = Y103.rotate(np.pi / 4, np.pi / 3, -np.pi / 8)
    N = 12
    nq = 48
    xg, wg = leggauss(nq)
    thq = np.arccos(xg)
    lamq = -np.pi + 2 * np.pi * np.arange(2 * nq) / (2 * nq)
    LQ, TQ = np.meshgrid(lamq, thq)
    GV = np.asarray(g3(LQ.ravel(), TQ.ravel())).reshape(LQ.shape)
    wl = 2 * np.pi / (2 * nq)
    C, Ls, Ms = [], [], []
    for el in range(N + 1):
        for m in range(-el, el + 1):
            YV = np.asarray(_real_ylm_values(
                el, m, LQ.ravel(), TQ.ravel())).reshape(LQ.shape)
            C.append(float(np.sum(GV * YV * wg[:, None]) * wl))
            Ls.append(el)
            Ms.append(m)
    C = np.array(C)
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.0, 6.0),
                           subplot_kw={"projection": "3d"})
    for cv, lv, mv in zip(np.abs(C), Ls, Ms):
        ax.plot([lv, lv], [mv, mv], [1e-16, max(cv, 1e-16)], 'b-',
                lw=0.8)
        ax.plot([lv], [mv], [max(cv, 1e-16)], 'bo', markersize=3)
    ax.set_zscale('log')
    ax.set_ylim(-N, N)
    ax.view_init(18, 167)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("m")
    ax.set_zlabel("|coeffs|")
    ax.set_title("Spherical harmonic coefficients")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"SpherefunRotate_{FIG[0]:02d}.png"),
             size=(600, 269))
    plt.close(fig)

    # Reconstruct from the degree-10 shell only.
    k0 = 10**2   # index where l=10 block starts
    def h_ev(lam_, th_):
        out = np.zeros(np.shape(np.asarray(lam_)))
        for j, m in enumerate(range(-10, 11)):
            out = out + C[k0 + j] * np.asarray(
                _real_ylm_values(10, m, lam_, th_))
        return out

    hrec = Spherefun.from_function(h_ev)
    print("ans =")
    print(f"     {float((hrec - g3).norm()):.15e}")

    # The NUFFT-based rotate versus naive evaluation ('feval') for a
    # random function with wavelength 0.03 (degree 209).
    f6, _ = _randnfunsphere(0.03, np.random.default_rng(0))
    _panel([(f6, "Random function")], "SpherefunRotate_05.png", n=450)
    t0 = time.perf_counter()
    f6.rotate(np.pi / 3, np.pi / 2, 0.5, "feval")
    t_feval = time.perf_counter() - t0
    t0 = time.perf_counter()
    h6 = f6.rotate(np.pi / 3, np.pi / 2, 0.5)
    t_nufft = time.perf_counter() - t0
    print(f"NUFFT speed-up factor = {t_feval / t_nufft:.1f}")
    _panel([(h6, "Rotated random function")], "SpherefunRotate_06.png",
           n=450)

    # Rank growth of rotated high-rank functions.
    f4 = Spherefun.from_function(
        lambda lam, th: np.cos(
            100 * (np.cos(lam) * np.sin(th))
            * (np.sin(lam) * np.sin(th))))
    print("ans =")
    print(f"    {f4.rank}")
    g4 = f4.rotate(0.01, 0.01, 0.01)
    print("ans =")
    print(f"    {g4.rank}")
    g5 = f4.rotate(np.pi / 4, -np.pi / 3, -np.pi / 8)
    print("ans =")
    print(f"   {g5.rank}")

    # Rank of a rotated Gaussian as it passes over the poles.
    cntr = -np.array([np.sin(0.1), np.cos(0.1)])
    f5 = Spherefun.from_function(
        lambda lam, th: np.exp(-10 * (
            (np.cos(lam) * np.sin(th) - cntr[0])**2
            + (np.sin(lam) * np.sin(th) - cntr[1])**2
            + np.cos(th)**2)))
    alp = np.linspace(0, 2 * np.pi, 101)
    rk = [f5.rotate(0, a, 0).rank for a in alp]
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.0, 2.69))
    ax.plot(alp, rk, 'x-', lw=2)
    ax.set_xlabel("Rotation angle")
    ax.set_ylabel("Rank")
    ax.set_title("Rank of the rotation of a Gaussian")
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"SpherefunRotate_{FIG[0]:02d}.png"),
             size=(600, 269))
    plt.close(fig)

if __name__ == "__main__":
    run()
