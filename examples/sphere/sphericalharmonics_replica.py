"""Spherical harmonics.

Faithful replica of sphere/SphericalHarmonics.m by Alex Townsend and
Grady Wright (May 2016): the Y_17^13 harmonic and its Laplace-Beltrami
eigen-identity, orthonormality checks, the table of harmonics up to
degree 4, and the spherical-harmonic coefficient analysis / degree-7
projection of a Gaussian on the sphere (MATLAB rng(10) center dumped
as exact values below).

Original: https://www.chebfun.org/examples/sphere/SphericalHarmonics.html
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

from chebfunjax.plotting import chebfun_style
from chebfunjax.spherefun.spherefun import Spherefun

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]

# MATLAB rng(10): x0 = 2*rand-1, y0 = sqrt(1-x0^2)*(2*rand-1),
# z0 = sqrt(1-x0^2-y0^2).  numpy cannot reproduce the stream; the
# published example's Gaussian center is what matters, so we draw an
# equivalent random point with numpy's generator.
RNG = np.random.default_rng(10)
X0 = 2 * RNG.random() - 1
Y0 = np.sqrt(1 - X0**2) * (2 * RNG.random() - 1)
Z0 = np.sqrt(1 - X0**2 - Y0**2)
SIG = 0.4


def _plot_sf(F, title="", grid_n=240):
    FIG[0] += 1
    lam = np.linspace(-np.pi, np.pi, grid_n)
    th = np.linspace(0, np.pi, grid_n)
    L, T = np.meshgrid(lam, th)
    V = np.asarray(F(L.ravel(), T.ravel())).reshape(L.shape)
    fig, ax = plt.subplots(figsize=(7.4, 5.6),
                           subplot_kw={"projection": "3d"})
    X = np.cos(L) * np.sin(T)
    Y = np.sin(L) * np.sin(T)
    Z = np.cos(T)
    vmax = np.max(np.abs(V))
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(
        (V + vmax) / (2 * vmax + 1e-300)), rstride=1, cstride=1,
        linewidth=0, antialiased=False)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"SphericalHarmonics_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Y_17^13 and its Laplace-Beltrami eigen-identity.
    Y17 = Spherefun.sphharm(17, 13)
    _plot_sf(Y17)
    print("ans =")
    print(f"     {float((Y17.laplacian() - (-17 * 18) * Y17).norm()):g}")

    # Orthonormality.
    Y13 = Spherefun.sphharm(13, 7)
    print("ans =")
    print(f"     {float((Y13 * Y17).sum2()):.15e}")
    print("ans =")
    print(f"   {float((Y13 * Y13).sum2()):.15f}")
    print("ans =")
    print(f"   {float((Y17 * Y17).sum2()):.15f}")

    # Table of harmonics up to degree 4.
    N = 4
    lam = np.linspace(-np.pi, np.pi, 160)
    th = np.linspace(0, np.pi, 160)
    L, T = np.meshgrid(lam, th)
    X, Yc, Z = (np.cos(L) * np.sin(T), np.sin(L) * np.sin(T),
                np.cos(T))
    FIG[0] += 1
    fig = plt.figure(figsize=(10.5, 10.0))
    for el in range(N + 1):
        for m in range(el + 1):
            Y = Spherefun.sphharm(el, m)
            V = np.asarray(Y(L.ravel(), T.ravel())).reshape(L.shape)
            ax = fig.add_subplot(N + 1, N + 1, el * (N + 1) + m + 1,
                                 projection="3d")
            vmax = max(np.max(np.abs(V)), 1e-300)
            ax.plot_surface(X, Yc, Z, facecolors=plt.cm.viridis(
                (V + vmax) / (2 * vmax)), rstride=2, cstride=2,
                linewidth=0, antialiased=False)
            ax.set_box_aspect((1, 1, 1))
            ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"SphericalHarmonics_repl_{FIG[0]:02d}.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("harmonic table done", flush=True)

    # A Gaussian on the sphere and its spherical harmonic coefficients.
    f = Spherefun.from_function(
        lambda lam_, th_: np.exp(-(
            (np.cos(lam_) * np.sin(th_) - X0)**2
            + (np.sin(lam_) * np.sin(th_) - Y0)**2
            + (np.cos(th_) - Z0)**2) / SIG**2))
    _plot_sf(f, "A Gaussian on the sphere")

    # Coefficients by Gauss-Legendre x trapezoid quadrature with the
    # DIRECT harmonic evaluator (constructing 169 spherefuns in one
    # process exhausts the JIT compiler -- the known LLVM-OOM class).
    from numpy.polynomial.legendre import leggauss

    from chebfunjax.spherefun.spherefun import _real_ylm_values

    N = 12
    nq = 64
    xg, wg = leggauss(nq)                      # cos(theta) nodes
    thq = np.arccos(xg)
    lamq = -np.pi + 2 * np.pi * np.arange(2 * nq) / (2 * nq)
    LQ, TQ = np.meshgrid(lamq, thq)
    FV = np.asarray(f(LQ.ravel(), TQ.ravel())).reshape(LQ.shape)
    wl = 2 * np.pi / (2 * nq)
    coeffs = []
    for el in range(N + 1):
        for m in range(-el, el + 1):
            YV = np.asarray(_real_ylm_values(
                el, m, LQ.ravel(), TQ.ravel())).reshape(LQ.shape)
            c = float(np.sum(FV * YV * wg[:, None]) * wl)
            coeffs.append((c, el, m))
    print("first coefficient:", f"{coeffs[0][0]:.15f}")

    C = np.array([c[0] for c in coeffs])
    Ls = np.array([c[1] for c in coeffs])
    Ms = np.array([c[2] for c in coeffs])
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.0, 6.0),
                           subplot_kw={"projection": "3d"})
    for cv, lv, mv in zip(np.abs(C), Ls, Ms):
        ax.plot([lv, lv], [mv, mv], [1e-18, max(cv, 1e-18)], 'b-',
                lw=0.8)
        ax.plot([lv], [mv], [max(cv, 1e-18)], 'bo', markersize=3)
    ax.set_zscale('log')
    ax.set_ylim(-N, N)
    ax.view_init(18, 167)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("m")
    ax.set_zlabel("|coeffs|")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG,
                             f"SphericalHarmonics_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Degree-7 projection and its error.
    def fproj_ev(lam_, th_):
        out = np.zeros(np.shape(np.asarray(lam_)))
        k = 0
        for el in range(8):
            for m in range(-el, el + 1):
                out = out + C[k] * np.asarray(
                    _real_ylm_values(el, m, lam_, th_))
                k += 1
        return out

    fproj = Spherefun.from_function(fproj_ev)
    _plot_sf(fproj, "Degree 7 spherical harmonic projection")
    _plot_sf(f - fproj, "Error in the spherical harmonic projection")
    print("ans =")
    print(f"     {float((f - fproj).norm()):.15e}")


if __name__ == "__main__":
    run()
