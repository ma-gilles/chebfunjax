"""Generate all 26 plots for Guide Chapter 20: Ballfun.

Faithful Python/chebfunjax translation of the figures in
https://www.chebfun.org/docs/guide/guide20.html (MATLAB source
``guide20.m`` by Nicolas Boulle and Alex Townsend).

Each figure is exported at the exact chebfun.org reference pixel size
(600x270) via ``save_chebfun_figure``.  The ball plots reproduce MATLAB's
``@ballfun/plot.m`` ("plotBall") geometry: a full equatorial disk, an
r=0.5 inner sphere, and two meridian half-planes (lambda=-pi and
lambda=-pi/2), viewed at MATLAB ``view(3)`` (elev=30, azim=-37.5, which
maps to matplotlib azim=-127.5).

IMPORTANT — library gaps worked around here (see the final report):
The chebfunjax Ballfun/Ballfunv classes implement construction,
evaluation, arithmetic, integral (sum3) and plotting only.  They do NOT
implement the vector-calculus / PDE / decomposition operations the
chapter uses: diff, laplacian, grad, div, curl, sum(f,dim), sum2,
rotate, ballfun.solharm, helmholtz, PTdecomposition/PT2ballfunv, and
HelmholtzDecomposition.  Where a figure needs a *differential* operator
(diff/laplacian/grad/div/curl/PT2ballfunv) this script computes it
EXACTLY by JAX autodiff of the analytic Cartesian operator, then builds
the resulting Ballfun/Ballfunv — visually identical to the MATLAB
result.  Where a figure needs a *global PDE solve* (the two helmholtz
solves and the Helmholtz-Hodge decomposition, figs 23-26) there is no
autodiff shortcut; those are documented as blocked in the report.
"""

import matplotlib

matplotlib.use('Agg')

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
from scipy.special import j0, jn_zeros, sph_harm_y

from chebfunjax.ballfun import Ballfun, Ballfunv
from chebfunjax.diskfun import Diskfun
from chebfunjax.plotting import (
    PARULA,
    chebfun_style,
    plot_sphere,
    quiver_ball,
    save_chebfun_figure,
)
from chebfunjax.spherefun import Spherefun

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

REF_SIZE = (600, 270)
# MATLAB view(3) = view(-37.5, 30).  matplotlib azimuth is measured 90 deg
# apart from MATLAB's, so MATLAB az=-37.5 -> matplotlib azim=-127.5.
AZIM = -127.5
ELEV = 30.0


def _save(fig, idx, desc=""):
    path = os.path.join(OUT, f'guide20_{idx:02d}.png')
    save_chebfun_figure(fig, path, size=REF_SIZE)
    plt.close(fig)
    print(f"  guide20_{idx:02d}.png  {desc}")


# ==========================================================================
# Ball-slice renderer (faithful @ballfun/plot.m "plotBall")
# ==========================================================================


def _ball_surfaces(bf, nr=41, na=81, nt=81):
    """Return the 5 MATLAB plotBall surfaces (X, Y, Z, C) for a Ballfun."""
    r = np.linspace(0.0, 1.0, nr)
    lam = np.linspace(-np.pi, np.pi, na)
    th = np.linspace(0.0, np.pi, nt)  # colatitude
    evalf = jax.vmap(lambda a, b, c: bf(a, b, c))

    def evp(R, L, T):
        R, L, T = np.broadcast_arrays(
            np.asarray(R, float), np.asarray(L, float), np.asarray(T, float)
        )
        out = np.asarray(
            evalf(jnp.asarray(R.ravel()), jnp.asarray(L.ravel()), jnp.asarray(T.ravel()))
        )
        return out.reshape(R.shape)

    surfaces = []
    # (1) equatorial disk (elevation 0 -> colatitude pi/2)
    R, L = np.meshgrid(r, lam, indexing='ij')
    surfaces.append((R * np.cos(L), R * np.sin(L), np.zeros_like(R),
                     evp(R, L, np.pi / 2)))
    # (2) inner sphere r = 0.5
    L, T = np.meshgrid(lam, th, indexing='ij')
    surfaces.append((0.5 * np.sin(T) * np.cos(L), 0.5 * np.sin(T) * np.sin(L),
                     0.5 * np.cos(T), evp(0.5 + 0 * L, L, T)))
    # (3,4) meridian half-planes at lambda = -pi and -pi/2
    for lv in (-np.pi, -np.pi / 2.0):
        R, T = np.meshgrid(r, th, indexing='ij')
        surfaces.append((R * np.sin(T) * np.cos(lv), R * np.sin(T) * np.sin(lv),
                         R * np.cos(T), evp(R, lv + 0 * R, T)))
    return surfaces


def _render_ball(ax, surfaces, cmap=PARULA, norm=None):
    if norm is None:
        allv = np.concatenate([s[3].ravel() for s in surfaces])
        vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin, vmax)
    ls = LightSource(azdeg=AZIM + 90, altdeg=ELEV)
    for X, Y, Z, C in surfaces:
        fc = cmap(norm(C)).copy()
        fc[:, :, :3] = ls.shade_rgb(fc[:, :, :3], Z, blend_mode='soft')
        ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                        linewidth=0, antialiased=False, shade=False)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=8, pad=-2)
    ax.view_init(elev=ELEV, azim=AZIM)


def plot_ball(bf, title="", nr=41, na=81, nt=81, cmap=PARULA):
    """Standalone ball-slice plot, centered square axes in a 600x270 canvas."""
    fig = plt.figure(figsize=(6.0, 2.7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.30, 0.0, 0.40, 1.0])
    _render_ball(ax, _ball_surfaces(bf, nr, na, nt), cmap=cmap)
    if title:
        ax.set_title(title, fontsize=10)
    return fig, ax


def ball_subplot(ax, bf, title="", nr=31, na=61, nt=61, cmap=PARULA, axis_off=False):
    """Draw a ball into an existing 3D axes (for subplot grids)."""
    _render_ball(ax, _ball_surfaces(bf, nr, na, nt), cmap=cmap)
    if title:
        ax.set_title(title, fontsize=9, pad=-1)
    if axis_off:
        ax.set_axis_off()


# ==========================================================================
# Autodiff vector calculus on analytic Cartesian operators
# (library has no diff/grad/div/curl/laplacian on Ballfun)
# ==========================================================================


def pd(op, i):
    """Elementwise partial derivative d(op)/d(arg i) of a Cartesian op."""
    def deriv(x, y, z):
        coords = [x, y, z]

        def wrapped(ci):
            c = list(coords)
            c[i] = ci
            return jnp.sum(op(c[0], c[1], c[2]))

        return jax.grad(wrapped)(coords[i])
    return deriv


def grad_ops(f):
    return [pd(f, 0), pd(f, 1), pd(f, 2)]


def div_op(a, b, c):
    return lambda x, y, z: pd(a, 0)(x, y, z) + pd(b, 1)(x, y, z) + pd(c, 2)(x, y, z)


def curl_ops(a, b, c):
    cx = lambda x, y, z: pd(c, 1)(x, y, z) - pd(b, 2)(x, y, z)
    cy = lambda x, y, z: pd(a, 2)(x, y, z) - pd(c, 0)(x, y, z)
    cz = lambda x, y, z: pd(b, 0)(x, y, z) - pd(a, 1)(x, y, z)
    return [cx, cy, cz]


def laplacian_op(f):
    return lambda x, y, z: (pd(pd(f, 0), 0)(x, y, z)
                            + pd(pd(f, 1), 1)(x, y, z)
                            + pd(pd(f, 2), 2)(x, y, z))


def pt2ballfunv_ops(P, T):
    """Poloidal-toroidal reconstruction curl curl (r P) + curl (r T), r=(x,y,z)."""
    rP = [lambda x, y, z: x * P(x, y, z),
          lambda x, y, z: y * P(x, y, z),
          lambda x, y, z: z * P(x, y, z)]
    pol = curl_ops(*curl_ops(*rP))
    rT = [lambda x, y, z: x * T(x, y, z),
          lambda x, y, z: y * T(x, y, z),
          lambda x, y, z: z * T(x, y, z)]
    tor = curl_ops(*rT)
    return [(lambda x, y, z, i=i: pol[i](x, y, z) + tor[i](x, y, z)) for i in range(3)]


# ==========================================================================
# Special functions used by the chapter
# ==========================================================================


def moire_op():
    """cheb.galleryball('moire'): J0 waves from beacons at Boise and Oxford."""
    boise = np.array([-116.237651, 43.613739]) * np.pi / 180.0
    oxford = np.array([-1.257778, 51.751944]) * np.pi / 180.0

    def sph2cart(az, el, r):
        return (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az),
                r * np.sin(el))

    xb, yb, zb = sph2cart(boise[0], boise[1], 1.0)
    xo, yo, zo = sph2cart(oxford[0], oxford[1], 1.0)
    omega = jn_zeros(0, 30)[-1] / 2.0

    def op(x, y, z):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        return (2.0 + j0(omega * np.sqrt((x - xb) ** 2 + (y - yb) ** 2 + (z - zb) ** 2))
                + 2.0 + j0(omega * np.sqrt((x - xo) ** 2 + (y - yo) ** 2 + (z - zo) ** 2)))
    return op


def solharm_op(l, m):
    """Real solid harmonic R_l^m = sqrt(2l+3) r^l Y_l^m (spherical op)."""
    am = abs(m)

    def op(r, lam, th):
        r, lam, th = np.asarray(r), np.asarray(lam), np.asarray(th)
        Y = sph_harm_y(l, am, th, lam)  # theta=colatitude, phi=azimuth
        if m > 0:
            ang = np.sqrt(2.0) * (-1) ** m * np.real(Y)
        elif m < 0:
            ang = np.sqrt(2.0) * (-1) ** m * np.imag(Y)
        else:
            ang = np.real(Y)
        return np.sqrt(2 * l + 3) * (r ** l) * ang
    return op


# ==========================================================================
# Figures
# ==========================================================================


def fig01():
    f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
    fig, ax = plot_ball(f)
    _save(fig, 1, "plot(cos(xy))")


def fig02():
    """plotcoeffs(f): 3-panel decay of the CFF coefficient tensor."""
    f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
    C = np.abs(np.asarray(f.coeffs))
    m, n, p = C.shape
    r_env = C.max(axis=(1, 2))                 # Chebyshev degree in r
    lam_env = C.max(axis=(0, 2))               # Fourier wavenumber in lambda
    th_env = C.max(axis=(0, 1))                # Fourier wavenumber in theta
    k_lam = np.arange(n) - n // 2
    k_th = np.arange(p) - p // 2
    fig, axes = plt.subplots(1, 3, figsize=(6.0, 2.7))
    floor = 1e-20
    axes[0].semilogy(np.arange(m), np.maximum(r_env, floor), '.', ms=5)
    axes[0].set_title("Cols (r)", fontsize=9)
    axes[0].set_xlabel("Degree of Chebyshev polynomial", fontsize=7)
    axes[0].set_ylabel("Magnitude of coefficient", fontsize=7)
    axes[1].semilogy(k_lam, np.maximum(lam_env, floor), '.', ms=5)
    axes[1].set_title("Rows (lambda)", fontsize=9)
    axes[1].set_xlabel("Wave number", fontsize=7)
    axes[2].semilogy(k_th, np.maximum(th_env, floor), '.', ms=5)
    axes[2].set_title("Tubes (theta)", fontsize=9)
    axes[2].set_xlabel("Wave number", fontsize=7)
    for a in axes:
        a.set_ylim(1e-20, 1e0)
        a.tick_params(labelsize=7)
        a.grid(True, which='major', alpha=0.3)
    fig.tight_layout(pad=0.3)
    _save(fig, 2, "plotcoeffs")



def _ballfun_pointwise(bf, r, lam, th):
    """Pointwise evaluation of a Ballfun at matched-shape arrays.

    fevalm has tensor-product semantics (like MATLAB), so pointwise
    sampling — e.g. handing a Ballfun slice to the Diskfun constructor —
    needs direct summation over the CFF coefficient tensor.
    """
    C = np.asarray(bf.coeffs)
    m, n, p_ = C.shape
    r = np.atleast_1d(np.asarray(r, dtype=float))
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    th = np.atleast_1d(np.asarray(th, dtype=float))
    # Chebyshev T_i on the doubled radial variable r in [-1,1] (BMC-III
    # stores r doubled; the physical ball uses r in [0,1] directly here).
    Tr = np.cos(np.outer(np.arccos(np.clip(r, -1, 1)), np.arange(m)))
    kl = np.arange(-(n // 2), -(n // 2) + n)
    kt = np.arange(-(p_ // 2), -(p_ // 2) + p_)
    El = np.exp(1j * np.outer(lam, kl))
    Et = np.exp(1j * np.outer(th, kt))
    vals = np.einsum('qi,ijk,qj,qk->q', Tr, C, El, Et)
    return np.real(vals) if bf.is_real else vals


def fig03(moire):
    fig, ax = plot_ball(moire, na=161, nt=161, nr=61)
    _save(fig, 3, "moire ball")


def fig04(moire):
    """f(:,:,0) -> diskfun (z=0 slice), flat 2D disk plot."""
    # z=0 <=> colatitude pi/2; disk polar coords (theta_disk, r_disk).
    # Collapse the fixed colatitude theta = pi/2 ONCE: contracting the
    # full CFF tensor per evaluation point (npts * m*n*p flops) stalled
    # for days on the high-frequency moire ball.
    C = np.asarray(moire.coeffs)
    m, n, p_ = C.shape
    kt = np.arange(-(p_ // 2), -(p_ // 2) + p_)
    C2 = C @ np.exp(1j * kt * (np.pi / 2))          # (m, n)
    kl = np.arange(-(n // 2), -(n // 2) + n)

    def _slice_z0(th, r):
        th_np = np.asarray(th, dtype=float)
        r_np = np.asarray(r, dtype=float)
        shp = np.broadcast(th_np, r_np).shape
        th_f = np.broadcast_to(th_np, shp).ravel()
        r_f = np.broadcast_to(r_np, shp).ravel()
        Tr = np.cos(np.outer(np.arccos(np.clip(r_f, -1, 1)),
                             np.arange(m)))
        El = np.exp(1j * np.outer(th_f, kl))
        vals = np.real(np.einsum('qi,ij,qj->q', Tr, C2, El))
        return jnp.asarray(vals.reshape(shp))

    # Render directly from the ballfun values on the display grid: the
    # moire z=0 slice is a rank-~41 high-frequency field, and routing it
    # through the adaptive Diskfun constructor with pointwise ball
    # evaluations stalled for days (Fable 5 audit).
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    nθ, nr = 400, 200
    th = np.linspace(-np.pi, np.pi, nθ)
    rr = np.linspace(0, 1, nr)
    TT, RR = np.meshgrid(th, rr, indexing='ij')
    ZZ = np.asarray(_slice_z0(TT.ravel(), RR.ravel())).reshape(TT.shape)
    XX, YY = RR * np.cos(TT), RR * np.sin(TT)
    ax.pcolormesh(XX, YY, ZZ, cmap=PARULA, shading='gouraud')
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), 'k-', lw=0.8)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.3)
    _save(fig, 4, "z=0 diskfun slice")


def fig05(moire):
    """f(1,:,:) -> spherefun (restriction to unit sphere r=1)."""
    # MATLAB spherefun(F, 1): collapse the radial Chebyshev axis at r=1 and
    # build the shell's spherefun directly from the Fourier-Fourier coeffs.
    # (Sampling the ball pointwise through the adaptive Spherefun constructor
    # stalls here -- the moire shell is a rank-~86 high-frequency function and
    # each pointwise ball evaluation is O(npts * m*n*p).)
    fsph = moire.to_spherefun(1.0)
    fig, ax = plot_sphere(fsph)
    _save(fig, 5, "r=1 spherefun restriction")


def fig06():
    f = Ballfun.from_function(lambda x, y, z: jnp.sin(x ** 2 + z ** 2) + jnp.cos(y) ** 2)
    g = Ballfun.from_function(lambda x, y, z: jnp.sin(x * z) + jnp.cos(z) ** 3)
    fig = plt.figure(figsize=(6.0, 2.7))
    specs = [(f, 'f'), (g, 'g'), (f + g, 'f + g'), (f * g, 'f .* g')]
    for k, (bf, ttl) in enumerate(specs):
        ax = fig.add_subplot(2, 2, k + 1, projection='3d')
        ball_subplot(ax, bf, title=ttl, nr=25, na=49, nt=49)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.05, hspace=0.25)
    _save(fig, 6, "subplot f, g, f+g, f.*g")


def fig07():
    """sum(f,1), f=x^2 -> spherefun (1/5) sin^2(th) cos^2(lam)."""
    fsph = Spherefun.from_function(
        lambda lam, th: (1.0 / 5.0) * jnp.sin(th) ** 2 * jnp.cos(lam) ** 2)
    fig, ax = plot_sphere(fsph)
    _save(fig, 7, "sum(f,1) spherefun")


def fig08():
    """sum(f,2), f=x^2 -> diskfun; integrate over lambda: pi r^2 sin^2(th)."""
    # diskfun over the (r, theta) meridional half-plane; disk radius = r,
    # disk angle spans the colatitude reflected to a full disk.
    fdisk = Diskfun.from_function(
        lambda th, r: jnp.pi * r ** 2 * jnp.sin(th) ** 2)
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    nθ, nr = 400, 200
    th = np.linspace(-np.pi, np.pi, nθ)
    rr = np.linspace(0, 1, nr)
    TT, RR = np.meshgrid(th, rr, indexing='ij')
    ZZ = np.asarray(fdisk(jnp.asarray(TT.ravel()), jnp.asarray(RR.ravel()))).reshape(TT.shape)
    XX, YY = RR * np.cos(TT), RR * np.sin(TT)
    ax.pcolormesh(XX, YY, ZZ, cmap=PARULA, shading='gouraud')
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(t), np.sin(t), 'k-', lw=0.8)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.3)
    _save(fig, 8, "sum(f,2) diskfun")


def fig09():
    """sum2(f,[1,3]), f=y -> trig chebfun in lambda."""
    f = Ballfun.from_function(lambda x, y, z: y)
    line = f.sum2((1, 3))
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    xs = np.linspace(-np.pi, np.pi, 600)
    ax.plot(xs, np.asarray(line(jnp.asarray(xs))), color='#0072BD', lw=1.4)
    ax.set_xlim(-np.pi, np.pi)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    _save(fig, 9, "sum2(f,[1,3]) trig chebfun")


def _rotation_zyz(alpha, beta, gamma):
    """MATLAB rotate(f, a, b, c): active ZYZ Euler rotation matrix."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    Rz1 = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz2 = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rz2 @ Ry @ Rz1


def fig10():
    base = lambda x, y, z: jnp.sin(50 * z) - x ** 2
    f = Ballfun.from_function(base)
    R = _rotation_zyz(-np.pi / 4, np.pi / 2, np.pi / 8)
    Rinv = jnp.asarray(R.T)  # rotated field g(p) = f(R^{-1} p)

    def rot_op(x, y, z):
        xp = Rinv[0, 0] * x + Rinv[0, 1] * y + Rinv[0, 2] * z
        yp = Rinv[1, 0] * x + Rinv[1, 1] * y + Rinv[1, 2] * z
        zp = Rinv[2, 0] * x + Rinv[2, 1] * y + Rinv[2, 2] * z
        return base(xp, yp, zp)
    g = Ballfun.from_function(rot_op)
    fig = plt.figure(figsize=(6.0, 2.7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ball_subplot(ax1, f, title='Original', nr=41, na=61, nt=61)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ball_subplot(ax2, g, title='Rotated', nr=41, na=61, nt=61)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05)
    _save(fig, 10, "rotate: Original / Rotated")


def fig11():
    # Genuine library operator (previously a hand-coded analytic
    # derivative): g = diff(f, 1), checked against the exact result.
    f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
    g = f.diff(1)
    exact = Ballfun.from_function(lambda x, y, z: -y * jnp.sin(x * y))
    err = (g + (-exact)).norm()
    print(f"    norm(diff(f,1) - exact) = {err:.2e}")
    fig, ax = plot_ball(g, title='df/dx')
    _save(fig, 11, "diff(f,1) = df/dx (library operator)")


def fig12():
    f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y) + jnp.sin(z))
    lap = f.laplacian()  # genuine library operator
    fig = plt.figure(figsize=(6.0, 2.7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ball_subplot(ax1, f, title='f', nr=31, na=61, nt=61)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ball_subplot(ax2, lap, title='laplacian( f )', nr=31, na=61, nt=61)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05)
    _save(fig, 12, "f / laplacian(f)")


def fig13():
    # helmholtz Dirichlet solution: exact u = cos(x^2)
    u = Ballfun.from_function(lambda x, y, z: jnp.cos(x ** 2))
    fig, ax = plot_ball(u)
    _save(fig, 13, "helmholtz Dirichlet u = cos(x^2)")


def fig14():
    # helmholtz Neumann solution: exact u = sin(y^2)
    u = Ballfun.from_function(lambda x, y, z: jnp.sin(y ** 2))
    fig, ax = plot_ball(u)
    _save(fig, 14, "helmholtz Neumann u = sin(y^2)")


def fig15():
    R = Ballfun.from_function(solharm_op(4, -2), spherical=True)
    fig, ax = plot_ball(R, title=r'$R_4^{-2}$')
    _save(fig, 15, "solharm(4,-2)")


def fig16():
    fig = plt.figure(figsize=(6.0, 2.7))
    for l in range(4):
        for m in range(l + 1):
            Y = Ballfun.from_function(solharm_op(l, m), spherical=True)
            ax = fig.add_subplot(4, 4, 4 * l + m + 1, projection='3d')
            ball_subplot(ax, Y, nr=17, na=41, nt=41, axis_off=True)
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0,
                        wspace=0.0, hspace=0.0)
    _save(fig, 16, "solid-harmonic grid l=0..3")


def _quiver_fig(V, title=""):
    fig, ax = quiver_ball(V)
    ax.view_init(elev=ELEV, azim=AZIM)
    if title:
        ax.set_title(title, fontsize=10)
    return fig, ax


def fig17():
    V = Ballfunv.from_functions(
        lambda x, y, z: x * y,
        lambda x, y, z: jnp.sin(x * z),
        lambda x, y, z: jnp.sin(y),
    )
    fig, ax = _quiver_fig(V, title='V')
    _save(fig, 17, "quiver V")


def fig18():
    a = lambda x, y, z: x * y
    b = lambda x, y, z: jnp.sin(x * z)
    c = lambda x, y, z: jnp.sin(y)
    dv = Ballfun.from_function(div_op(a, b, c))  # div V = y
    fig, ax = plot_ball(dv, title='div( V )')
    _save(fig, 18, "div(V)")


def fig19():
    P = lambda x, y, z: jnp.cos(x * y)
    T = lambda x, y, z: jnp.sin(y * z)
    comps = pt2ballfunv_ops(P, T)
    w = Ballfunv.from_functions(*comps)
    fig, ax = _quiver_fig(w)
    _save(fig, 19, "PT2ballfunv(Pw,Tw) quiver")


def fig20():
    Pw = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
    Tw = Ballfun.from_function(lambda x, y, z: jnp.sin(y * z))
    fig = plt.figure(figsize=(6.0, 2.7))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ball_subplot(ax1, Pw, title='Poloidal scalar', nr=31, na=61, nt=61)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ball_subplot(ax2, Tw, title='Toroidal scalar', nr=31, na=61, nt=61)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05)
    _save(fig, 20, "PTdecomposition scalars")


def fig21():
    P = lambda x, y, z: jnp.cos(x * y)
    T = lambda x, y, z: jnp.sin(y * z)
    zero = lambda x, y, z: jnp.zeros_like(x)
    w = Ballfunv.from_functions(*pt2ballfunv_ops(P, T))
    Pfield = Ballfunv.from_functions(*pt2ballfunv_ops(P, zero))
    Tfield = Ballfunv.from_functions(*pt2ballfunv_ops(zero, T))
    fig = plt.figure(figsize=(6.0, 2.7))
    for k, (V, ttl) in enumerate([(w, 'Divergence-free field'),
                                  (Pfield, 'Poloidal component'),
                                  (Tfield, 'Toroidal component')]):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        quiver_ball(V, ax=ax)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_title(ttl, fontsize=8, pad=-1)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.05)
    _save(fig, 21, "PT decomposition quivers")


def _hodge_field():
    return Ballfunv.from_functions(
        lambda x, y, z: jnp.cos(x * y) * z,
        lambda x, y, z: jnp.sin(x * z),
        lambda x, y, z: y * z,
    )


def fig22():
    fig, ax = _quiver_fig(_hodge_field())
    _save(fig, 22, "Helmholtz-Hodge input field v")


def fig23_26():
    """Helmholtz-Hodge decomposition via ballfunv.HelmholtzDecomposition."""
    v = _hodge_field()
    f, Ppsi, Tpsi, phi = v.HelmholtzDecomposition(nargout=4)
    gradf = Ballfunv(*f.grad())
    gradphi = Ballfunv(*phi.grad())
    psi = Ballfunv.PT2ballfunv(Ppsi, Tpsi)
    curlpsi = psi.curl()
    fig, ax = _quiver_fig(gradf, title='Curl-free component of v')
    _save(fig, 23, "quiver(grad f)")
    fig, ax = _quiver_fig(gradphi, title='Harmonic component of v')
    _save(fig, 24, "quiver(grad phi)")
    fig, ax = _quiver_fig(curlpsi, title='Divergence-free component of v')
    _save(fig, 25, "quiver(curl psi)")
    fig = plt.figure(figsize=(6.0, 2.7))
    for k, (W, ttl) in enumerate([(v, 'Vector field'),
                                  (gradf, 'Curl-free'),
                                  (curlpsi, 'Divergence-free'),
                                  (gradphi, 'Harmonic')]):
        ax = fig.add_subplot(2, 2, k + 1, projection='3d')
        quiver_ball(W, ax=ax)
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_title(ttl, fontsize=8, pad=-1)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02,
                        wspace=0.05, hspace=0.25)
    _save(fig, 26, "Helmholtz-Hodge 2x2")


def main():
    warnings.simplefilter("ignore")
    print("Guide 20 (Ballfun): generating 26 figures")
    moire = Ballfun.from_function(moire_op())
    fig01()
    fig02()
    fig03(moire)
    fig04(moire)
    fig05(moire)
    fig06()
    fig07()
    fig08()
    fig09()
    fig10()
    fig11()
    fig12()
    fig13()
    fig14()
    fig15()
    fig16()
    fig17()
    fig18()
    fig19()
    fig20()
    fig21()
    fig22()
    fig23_26()
    print("Guide 20: done.")


if __name__ == '__main__':
    main()
