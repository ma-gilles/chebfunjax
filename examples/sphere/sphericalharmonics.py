"""Spherical harmonics.

Translation of sphere/SphericalHarmonics.m by Alex Townsend and
Grady Wright (May 2016): the Y_17^13 harmonic and its Laplace-Beltrami
eigen-identity, orthonormality checks, the table of harmonics up to
degree 4, and the spherical-harmonic coefficient analysis / degree-7
projection of a Gaussian on the sphere, the Funk-Hecke formula, the
addition theorem, and the platonic-solid harmonic combinations.

Original: https://www.chebfun.org/examples/sphere/SphericalHarmonics.html
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
from numpy.polynomial.legendre import leggauss
from scipy.special import iv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun1d.chebfun import legpoly
from chebfunjax.plotting import CHEBFUN_BLUE, PARULA, chebfun_style, plot_sphere
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.spherefun.spherefun import Spherefun, _real_ylm_values

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'sphere')
FIG = [0]

# MATLAB rng(k), k >= 1, is the MT19937 stream of numpy's
# RandomState(k), so the published Gaussian center is reproduced:
# x0 = 2*rand-1, y0 = sqrt(1-x0^2)*(2*rand-1), z0 = sqrt(1-x0^2-y0^2).


def _rand_point(seed):
    rs = np.random.RandomState(seed)
    x0 = 2 * rs.random_sample() - 1
    y0 = np.sqrt(1 - x0**2) * (2 * rs.random_sample() - 1)
    return x0, y0, np.sqrt(1 - x0**2 - y0**2)


X0, Y0, Z0 = _rand_point(10)
SIG = 0.4


EL, AZ = 30, -37.5 - 90                   # MATLAB view(3)
EYE = np.array([np.cos(np.deg2rad(EL)) * np.cos(np.deg2rad(AZ)),
                np.cos(np.deg2rad(EL)) * np.sin(np.deg2rad(AZ)),
                np.sin(np.deg2rad(EL))])


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG,
                               f"SphericalHarmonics_{FIG[0]:02d}.png"))
    plt.close(fig)


def _contour0(F, ax, level=0.0, n=200):
    """hold on, contour(F, [level level], 'k-') on the visible side."""
    lam = np.linspace(-np.pi, np.pi, n)
    th = np.linspace(0, np.pi, n)
    L, T = np.meshgrid(lam, th)
    C = np.asarray(F(jnp.asarray(L.ravel()), jnp.asarray(T.ravel())))
    tmp = plt.figure()
    cs = tmp.add_subplot().contour(lam, th, C.reshape(L.shape), [level])
    plt.close(tmp)
    for seg in cs.allsegs[0]:
        P = 1.005 * np.stack([np.cos(seg[:, 0]) * np.sin(seg[:, 1]),
                              np.sin(seg[:, 0]) * np.sin(seg[:, 1]),
                              np.cos(seg[:, 1])], axis=1)
        P[P @ EYE < -1e-3] = np.nan
        ax.plot(P[:, 0], P[:, 1], P[:, 2], "k-", lw=0.8, zorder=10)


def _plot_sf(F, title="", colorbar=True, contour=None, ax=None):
    """plot(F), [hold on, contour(F, [c c], 'k-')], title, colorbar,
    axis off."""
    single = ax is None
    if single:
        fig = plt.figure(figsize=(6.0, 2.7))
        ax = fig.add_axes([0.15 if colorbar else 0.2, 0.0, 0.6, 0.9],
                          projection="3d")
    fig = ax.get_figure()
    plot_sphere(F, ax=ax)
    ax.view_init(elev=EL, azim=AZ)
    ax.set_axis_off()
    if contour is not None:
        _contour0(F, ax, contour)
    if title:
        ax.set_title(title, fontsize=10, pad=0)
    if single and colorbar:
        lam = np.linspace(-np.pi, np.pi, 200)
        L, T = np.meshgrid(lam, np.linspace(0, np.pi, 200))
        V = np.asarray(F(jnp.asarray(L.ravel()), jnp.asarray(T.ravel())))
        cax = fig.add_axes([0.8, 0.1, 0.025, 0.8])
        fig.colorbar(ScalarMappable(norm=Normalize(V.min(), V.max()),
                                    cmap=PARULA), cax=cax)
    if single:
        _save(fig)


def _stem3(C, Ls, Ms, N, zlim=None):
    """stem3(l, m, |c|, 'filled'), log z-scale, Xdir reverse,
    view([-13 18])."""
    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    lo = zlim[0] if zlim else 1e-18
    for cv, lv, mv in zip(np.abs(C), Ls, Ms):
        z = np.log10(max(cv, lo))
        ax.plot([lv, lv], [mv, mv], [np.log10(lo), z], '-',
                color=CHEBFUN_BLUE, lw=0.8)
        ax.plot([lv], [mv], [z], 'o', color=CHEBFUN_BLUE, markersize=3)
    zs = np.log10(np.maximum(np.abs(C), lo))
    zt = np.arange(5 * np.floor(zs.min() / 5), zs.max() + 1, 5)
    ax.set_zticks(zt)
    ax.tick_params(labelsize=8)
    ax.set_zticklabels([f"$10^{{{int(t)}}}$" for t in zt])
    ax.set_ylim(-N, N)
    ax.set_xlim(N, 0)                    # set(gca, 'Xdir', 'reverse')
    ax.view_init(18, -13 - 90)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("m")
    ax.set_zlabel("|coeffs|")
    return fig


def _ylm_sum(terms):
    """Spherefun of sum(c * Y_l^m) built from pointwise harmonic values
    (one construction instead of a Spherefun per harmonic)."""
    def ev(lam_, th_):
        out = 0.0
        for c, el, m in terms:
            out = out + c * _real_ylm_values(el, m, lam_, th_)
        return out
    return Spherefun.from_function(ev)


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

    # Table of harmonics up to degree 4 with their zero contours.
    N = 4
    fig = plt.figure(figsize=(6.0, 2.7))
    for el in range(N + 1):
        for m in range(el + 1):
            ax = fig.add_subplot(N + 1, N + 1, el * (N + 1) + m + 1,
                                 projection="3d")
            _plot_sf(Spherefun.sphharm(el, m), ax=ax, contour=0.0)
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    _save(fig)
    print("harmonic table done", flush=True)

    # 3. A Gaussian on the sphere and its spherical harmonic coefficients.
    f = Spherefun.from_function(
        lambda lam_, th_: jnp.exp(-(
            (jnp.cos(lam_) * jnp.sin(th_) - X0)**2
            + (jnp.sin(lam_) * jnp.sin(th_) - Y0)**2
            + (jnp.cos(th_) - Z0)**2) / SIG**2))
    print("f =")
    print("   spherefun object")
    print("       domain        rank    vertical scale")
    print(f"     unit sphere      {f.rank}          "
          f"{float(np.max(np.abs(np.asarray(f.sample(64, 64))))):.2g}")
    _plot_sf(f, "A Gaussian on the sphere")

    # sum2(f .* Y) by Gauss-Legendre x trapezoid quadrature with the
    # DIRECT harmonic evaluator (constructing 169 spherefuns in one
    # process exhausts the JIT compiler -- the known LLVM-OOM class).
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
    C = np.array([c[0] for c in coeffs])
    Ls = np.array([c[1] for c in coeffs])
    Ms = np.array([c[2] for c in coeffs])
    _save(_stem3(C, Ls, Ms, N))

    # Degree-7 projection and its error.
    fproj = _ylm_sum([c for c in coeffs if c[1] <= 7])
    _plot_sf(fproj, "Degree 7 spherical harmonic projection")
    _plot_sf(f - fproj, "Error in the spherical harmonic projection")
    print("ans =")
    print(f"   {float((f - fproj).norm()):.15f}")

    # 4. Zonal kernels and the Funk-Hecke formula.
    lam0 = np.arctan2(Y0, X0)
    th0 = np.arccos(Z0)
    Yx0 = np.array([float(_real_ylm_values(el, m, lam0, th0))
                    for el, m in zip(Ls, Ms)])
    _save(_stem3(C / Yx0, Ls, Ms, N))

    a = (np.sqrt(np.pi) / 2 * SIG * np.exp(-2 / SIG**2) * (2 * Ls + 1)
         * iv(Ls + 0.5, 2 / SIG**2))
    coeffsExact = 4 * np.pi / (2 * Ls + 1) * a * Yx0
    print("ans =")
    print(f"     {np.max(np.abs(C - coeffsExact)):.15e}")

    # 5. The Addition Theorem for l = 14.
    x0, y0, z0 = _rand_point(13)
    lam0, th0 = np.arctan2(y0, x0), np.arccos(z0)
    el = 14
    lhs = _ylm_sum([(4 * np.pi / (2 * el + 1)
                     * float(_real_ylm_values(el, m, lam0, th0)), el, m)
                    for m in range(-el, el + 1)])
    _plot_sf(lhs)

    p15 = legpoly(el)
    rhs = Spherefun.from_function(
        lambda lam_, th_: p15(jnp.clip(
            jnp.cos(lam_) * jnp.sin(th_) * x0
            + jnp.sin(lam_) * jnp.sin(th_) * y0 + jnp.cos(th_) * z0,
            -1.0, 1.0)))
    _plot_sf(rhs)
    print("ans =")
    print(f"     {float((lhs - rhs).norm()):.15e}")

    # 6. Platonic solids.
    Y = Spherefun.sphharm(3, 2)                    # tetrahedral
    _plot_sf(Y, colorbar=False, contour=0.1)
    Y = Spherefun.sphharm(4, 0) + np.sqrt(5 / 7) * Spherefun.sphharm(4, 4)
    _plot_sf(Y, colorbar=False, contour=0.0)       # octahedral
    Y = (Spherefun.sphharm(6, 0)
         + np.sqrt(14 / 11) * Spherefun.sphharm(6, 5))
    _plot_sf(Y, colorbar=False, contour=0.0)       # icosahedral


if __name__ == "__main__":
    run()
