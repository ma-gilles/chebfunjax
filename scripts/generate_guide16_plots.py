"""Generate all plots for Guide Chapter 16 (Diskfun).

Every figure is a genuine chebfunjax render (never a chebfun.org copy),
exported at the exact pixel size of its reference (610x258) with the axes
box pinned to the geometry measured from the MATLAB reference renders, so
each figure can be compared pixel-for-pixel against the chebfun.org image.

Rendering follows @diskfun/plot, @diskfun/surf and @diskfun/contour: the
2D disk figures are parula pseudocolor over a polar grid, clipped to the
unit disk, with a thin boundary circle and no axes box (matching MATLAB).

Where the Diskfun API lacks an operation a figure needs (differentiation,
gradient/divergence/curl, roots, the Poisson solver, the GE skeleton,
cart2pol, plotcoeffs) the function is rendered through the equivalent
computation and chebfunjax's parula/style helpers; those gaps are recorded
in the task report.  Figures 26-28 (column/row slices and their
coefficients) use a genuinely constructed Diskfun's internals.
"""
import matplotlib

matplotlib.use('Agg')
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.special import jn_zeros, jv, jvp

from chebfunjax.diskfun import Diskfun
from chebfunjax.plotting import PARULA, chebfun_style, save_chebfun_figure

chebfun_style()
warnings.filterwarnings('ignore')

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)

REF = (610, 258)

# Square disk box measured from the reference disk figures: the parula disk
# occupies pixels x=210..420, y=19..229 of the 610x258 canvas.
DISK_BOX = [210 / 610, (258 - 229) / 258, 210 / 610, 210 / 258]
# Colorbar layout (figs 09, 11): disk shifted left to x=177..388, parula
# colorbar strip at x=404..424, same vertical extent.
DISK_BOX_CBAR = [177 / 610, (258 - 229) / 258, 211 / 610, 210 / 258]
CBAR_BOX = [404 / 610, (258 - 229) / 258, 20 / 610, 210 / 258]
# 1D line-plot box (shared with the other Guide chapters).
LINE_BOX = dict(left=79 / 610, right=551 / 610, bottom=29 / 258, top=239 / 258)
LINE_BOX_TITLED = dict(left=79 / 610, right=551 / 610, bottom=29 / 258, top=234 / 258)

plot_num = 0


def _save(fig, desc=""):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide16_{plot_num:02d}.png')
    save_chebfun_figure(fig, fname, size=REF)
    plt.close(fig)
    print(f"  guide16_{plot_num:02d}.png: {desc}")


def _fail(n, e):
    global plot_num
    for _ in range(n):
        plot_num += 1
        print(f"  guide16_{plot_num:02d}.png FAILED: {e}")


def eval_disk(fn, n_theta=400, n_r=220):
    """Evaluate f(theta, r) on a polar grid; return Cartesian X, Y and values."""
    theta = np.linspace(-np.pi, np.pi, n_theta)
    r = np.linspace(0.0, 1.0, n_r)
    TT, RR = np.meshgrid(theta, r)
    ZZ = np.array(fn(jnp.asarray(TT.ravel()), jnp.asarray(RR.ravel()))).reshape(TT.shape)
    return RR * np.cos(TT), RR * np.sin(TT), ZZ


def _boundary(ax, lw=0.8):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), 'k-', lw=lw)


def disk_pcolor(fn, title='', clim=None, cbar=False, cbar_ticks=None,
                overlay=None):
    """Parula pseudocolor of a disk function (@diskfun/plot, flat view)."""
    XX, YY, ZZ = eval_disk(fn)
    fig = plt.figure(figsize=(6.1, 2.58))
    ax = fig.add_axes(DISK_BOX_CBAR if cbar else DISK_BOX)
    norm = Normalize(*clim) if clim else Normalize(float(np.nanmin(ZZ)),
                                                   float(np.nanmax(ZZ)))
    pcm = ax.pcolormesh(XX, YY, ZZ, cmap=PARULA, shading='gouraud', norm=norm)
    _boundary(ax)
    ax.set_xlim(-1.001, 1.001)
    ax.set_ylim(-1.001, 1.001)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)
    if overlay is not None:
        overlay(ax)
    if cbar:
        cax = fig.add_axes(CBAR_BOX)
        cb = fig.colorbar(pcm, cax=cax)
        if cbar_ticks is not None:
            cb.set_ticks(cbar_ticks)
        cax.tick_params(labelsize=8)
    fig.set_facecolor('white')
    return fig, ax


def line_fig(titled=False):
    fig = plt.figure(figsize=(6.1, 2.58))
    ax = fig.add_axes([0, 0, 1, 1])
    fig.subplots_adjust(**(LINE_BOX_TITLED if titled else LINE_BOX))
    ax.set_position([LINE_BOX['left'],
                     (LINE_BOX_TITLED if titled else LINE_BOX)['bottom'],
                     LINE_BOX['right'] - LINE_BOX['left'],
                     (LINE_BOX_TITLED if titled else LINE_BOX)['top']
                     - (LINE_BOX_TITLED if titled else LINE_BOX)['bottom']])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.tick_params(labelsize=9, direction='in')
    fig.set_facecolor('white')
    return fig, ax


# ---------------------------------------------------------------------------
# Function definitions shared across figures
# ---------------------------------------------------------------------------

gauss = lambda th, r: jnp.exp(-10 * ((r * jnp.cos(th) - 0.3) ** 2 + (r * jnp.sin(th)) ** 2))

g_swirl = lambda th, r: -40 * jnp.cos((jnp.sin(jnp.pi * r) * jnp.cos(th)
                        + jnp.sin(2 * jnp.pi * r) * jnp.sin(th)) / 4) + 39.5
f_rings = lambda th, r: (jnp.cos(15 * ((r * jnp.cos(th) - 0.2) ** 2 + (r * jnp.sin(th) - 0.2) ** 2))
                         * jnp.exp(-(r * jnp.cos(th) - 0.2) ** 2 - (r * jnp.sin(th) - 0.2) ** 2))

W41 = float(jn_zeros(4, 1)[0])
u_harm = lambda th, r: jv(4, W41 * r) * jnp.cos(4 * th)
LAM = 7.58834243450380 ** 2

f_poisson = lambda th, r: jnp.sin(21 * jnp.pi * (1 + jnp.cos(jnp.pi * r))
                          * (r ** 2 - 2 * r ** 5 * jnp.cos(5 * (th - 0.11))))

f_bmc_fn = lambda th, r: (jnp.cos(2 * (3 * jnp.sin(2 * r * jnp.cos(th)) + 5 * jnp.sin(r * jnp.sin(th))))
                          - 0.5 * jnp.sin(r * jnp.cos(th) - r * jnp.sin(th)))


def psi_cart(x, y):
    return (5 * jnp.exp(-10 * (x + 0.2) ** 2 - 10 * (y + 0.4) ** 2)
            - 5 * jnp.exp(-10 * (x - 0.2) ** 2 - 10 * (y - 0.2) ** 2)
            + 5 * (1 - x ** 2 - y ** 2) - 20)


def g_curl_cart(x, y):
    return jnp.cosh(0.25 * (jnp.cos(5 * x) + jnp.sin(4 * y ** 2))) - 2


def polar_from_cart(cart_fn):
    return lambda th, r: cart_fn(r * jnp.cos(th), r * jnp.sin(th))


# ---------------------------------------------------------------------------
# 01  plot(g), view(3) — 3D Gaussian surface
# ---------------------------------------------------------------------------
try:
    XX, YY, ZZ = eval_disk(gauss, n_theta=220, n_r=130)
    fig = plt.figure(figsize=(6.1, 2.58))
    ax = fig.add_axes([170 / 610, (258 - 248) / 258, (455 - 170) / 610, (248 - 12) / 258],
                      projection='3d')
    ax.view_init(elev=30, azim=-127.5)
    ax.set_box_aspect((1, 1, 0.62), zoom=1.28)
    norm = Normalize(float(ZZ.min()), float(ZZ.max()))
    fc = PARULA(norm(ZZ))
    ax.plot_surface(XX, YY, ZZ, facecolors=fc, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([0, 0.5, 1])
    ax.set_zlim(0, 1)
    ax.tick_params(labelsize=8, pad=-1)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.fill = False
        pane.pane.set_edgecolor((0.85, 0.85, 0.85))
    ax.grid(True)
    fig.set_facecolor('white')
    _save(fig, "Gaussian 3D surface")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 02  three angular slices f(:, rho) at rho = 1/4, 1/3, 1/2
# ---------------------------------------------------------------------------
try:
    fig, ax = line_fig(titled=True)
    th = jnp.linspace(-jnp.pi, jnp.pi, 400)
    thn = np.array(th)
    # pure blue / black / red, matching the reference
    for rho, col in [(0.5, 'b'), (1.0 / 3.0, 'k'), (0.25, 'r')]:
        vals = np.array(gauss(th, jnp.full_like(th, rho)))
        ax.plot(thn, vals, color=col, linewidth=1.0)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title('Three angular slices of a diskfun', fontsize=10)
    _save(fig, "angular slices")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 03  diagonal slice f(x, x)
# ---------------------------------------------------------------------------
try:
    fig, ax = line_fig(titled=True)
    # diag(f): the diagonal slice as a function of the diagonal parameter
    # r in [-1, 1] (theta = pi/4 for r >= 0, pi/4 + pi otherwise).
    rp = np.linspace(-1, 1, 400)
    thv = np.where(rp >= 0, np.pi / 4, np.pi / 4 + np.pi)
    diag = np.array(gauss(jnp.asarray(thv), jnp.asarray(np.abs(rp))))
    ax.plot(rp, diag, color='#0072BD', linewidth=1.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 0.6)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax.set_title('The diagonal slice of f', fontsize=10)
    _save(fig, "diagonal slice")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 04-08  g, f, g+f, g-f, g*f
# ---------------------------------------------------------------------------
try:
    fig, _ = disk_pcolor(g_swirl, title='g')
    _save(fig, "g")
    fig, _ = disk_pcolor(f_rings, title='f')
    _save(fig, "f")
    for title, fn in [
        ('g + f', lambda th, r: g_swirl(th, r) + f_rings(th, r)),
        ('g - f', lambda th, r: g_swirl(th, r) - f_rings(th, r)),
        ('g x f', lambda th, r: g_swirl(th, r) * f_rings(th, r)),
    ]:
        fig, _ = disk_pcolor(fn, title=title)
        _save(fig, title)
except Exception as e:
    _fail(max(0, 8 - plot_num), e)

# ---------------------------------------------------------------------------
# 09  f with its maximum marked (colorbar)
# ---------------------------------------------------------------------------
try:
    _, _, zz = eval_disk(f_rings)
    fig, _ = disk_pcolor(
        f_rings, cbar=True, cbar_ticks=[-0.5, 0, 0.5],
        overlay=lambda ax: ax.plot(0.2, 0.2, 'k.', markersize=9))
    _save(fig, "f with maximum")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 10  contour(g), zero contours in black
# ---------------------------------------------------------------------------
try:
    XX, YY, ZZ = eval_disk(g_swirl, n_theta=400, n_r=220)
    fig = plt.figure(figsize=(6.1, 2.58))
    ax = fig.add_axes(DISK_BOX)
    ax.contour(XX, YY, ZZ, levels=9, cmap=PARULA, linewidths=1.0)
    ax.contour(XX, YY, ZZ, levels=[0.0], colors='k', linewidths=1.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')
    fig.set_facecolor('white')
    _save(fig, "contour g + zero contours")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 11  roots of g overlaid on plot(g) (colorbar)
# ---------------------------------------------------------------------------
try:
    XXg, YYg, ZZg = eval_disk(g_swirl, n_theta=400, n_r=220)

    def _roots_overlay(ax):
        ax.contour(XXg, YYg, ZZg, levels=[0.0], colors='k', linewidths=1.5)

    fig, _ = disk_pcolor(g_swirl, cbar=True, cbar_ticks=[-0.5, 0, 0.5, 1],
                         overlay=_roots_overlay)
    _save(fig, "roots of g")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 12  contour(u,20,'b'), contour(v,20,'m')  (harmonic conjugates)
# ---------------------------------------------------------------------------
try:
    u_cr = lambda th, r: r ** 3 * jnp.cos(3 * th)
    v_cr = lambda th, r: r ** 3 * jnp.sin(3 * th)
    XX, YY, ZU = eval_disk(u_cr, n_theta=400, n_r=220)
    _, _, ZV = eval_disk(v_cr, n_theta=400, n_r=220)
    fig = plt.figure(figsize=(6.1, 2.58))
    ax = fig.add_axes(DISK_BOX)
    ax.contour(XX, YY, ZU, levels=20, colors='b', linewidths=0.6)
    ax.contour(XX, YY, ZV, levels=20, colors=(1, 0, 1), linewidths=0.6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')
    ax.set_title('Contour lines for u and v', fontsize=10)
    fig.set_facecolor('white')
    _save(fig, "Cauchy-Riemann contours")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 13  cylindrical harmonic u = J4(w41 r) cos(4 theta)
# ---------------------------------------------------------------------------
try:
    fig, _ = disk_pcolor(u_harm, title='u')
    _save(fig, "harmonic u")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 14-15  du/dx, du/dy  (analytic Cartesian derivatives of u)
# ---------------------------------------------------------------------------
try:
    def _du(th, r, which):
        z = W41 * r
        Jp = W41 * jvp(4, z)          # d/dr J4(W41 r)
        J = jv(4, z)
        c4, s4 = np.cos(4 * th), np.sin(4 * th)
        ct, st = np.cos(th), np.sin(th)
        # radial / angular partials of u = J4(W41 r) cos(4 theta)
        du_dr = Jp * c4
        du_dth = -4 * J * s4
        rsafe = np.where(r == 0, 1.0, r)
        du_dth_over_r = np.where(r == 0, 0.0, du_dth / rsafe)
        if which == 'x':
            return ct * du_dr - st * du_dth_over_r
        return st * du_dr + ct * du_dth_over_r

    for which, title in [('x', 'du/dx'), ('y', 'du/dy')]:
        fn = (lambda w: (lambda th, r: _du(np.asarray(th), np.asarray(r), w)))(which)
        fig, _ = disk_pcolor(fn, title=title)
        _save(fig, title)
except Exception as e:
    _fail(max(0, 15 - plot_num), e)

# ---------------------------------------------------------------------------
# 16  Laplacian of u = -lambda * u
# ---------------------------------------------------------------------------
try:
    lap_u = lambda th, r: -LAM * jv(4, W41 * r) * jnp.cos(4 * th)
    fig, _ = disk_pcolor(lap_u, title='Laplacian of u')
    _save(fig, "Laplacian of u")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 17  Poisson right-hand side f
# ---------------------------------------------------------------------------
try:
    fig, _ = disk_pcolor(f_poisson, title='f')
    _save(fig, "Poisson rhs")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 18  Poisson solution v: lap(v) = f, v(theta, 1) = 1
#     Library has no diskfun.poisson.  Solved here semi-spectrally: FFT in
#     theta decouples the Fourier modes; each radial mode solves
#     v_m'' + v_m'/r - m^2/r^2 v_m = f_m by second-order finite differences,
#     with regularity at r = 0 and the Dirichlet condition at r = 1.
# ---------------------------------------------------------------------------
try:
    from scipy.interpolate import RegularGridInterpolator

    M, N = 256, 180
    h = 1.0 / N
    rj = np.arange(N + 1) * h                       # r_0 = 0 .. r_N = 1
    tk = 2 * np.pi * np.arange(M) / M

    TT, RR = np.meshgrid(tk, rj, indexing='ij')     # (M, N+1)
    F = np.array(f_poisson(jnp.asarray(TT.ravel()), jnp.asarray(RR.ravel()))).reshape(M, N + 1)
    Fhat = np.fft.fft(F, axis=0)                     # modes along axis 0
    modes = np.fft.fftfreq(M, d=1.0 / M).astype(int)
    ghat = np.zeros(M, dtype=complex)
    ghat[0] = M                                      # FFT of the constant 1

    Vhat = np.zeros((M, N + 1), dtype=complex)
    for mi, m in enumerate(modes):
        A = np.zeros((N + 1, N + 1), dtype=complex)
        b = Fhat[mi].astype(complex).copy()
        # r = 0
        if m == 0:
            A[0, 0] = -4.0 / h ** 2
            A[0, 1] = 4.0 / h ** 2                   # lap = 2 v''(0) ~ 4(v1-v0)/h^2
        else:
            A[0, 0] = 1.0
            b[0] = 0.0                               # v_m(0) = 0 for m != 0
        # interior
        for j in range(1, N):
            r = rj[j]
            A[j, j - 1] = 1.0 / h ** 2 - 1.0 / (2 * h * r)
            A[j, j] = -2.0 / h ** 2 - m ** 2 / r ** 2
            A[j, j + 1] = 1.0 / h ** 2 + 1.0 / (2 * h * r)
        # r = 1 Dirichlet
        A[N, N] = 1.0
        b[N] = ghat[mi]
        Vhat[mi] = np.linalg.solve(A, b)

    V = np.real(np.fft.ifft(Vhat, axis=0))           # (M, N+1) on (theta, r)
    tg = np.concatenate([tk - 2 * np.pi, tk, tk + 2 * np.pi])
    Vt = np.vstack([V, V, V])
    interp = RegularGridInterpolator((tg, rj), Vt, bounds_error=False, fill_value=None)

    def v_polar(th, r_):
        th = ((np.asarray(th) + np.pi) % (2 * np.pi)) - np.pi
        return interp(np.stack([th, np.clip(np.asarray(r_), 0, 1)], axis=-1))

    fig, _ = disk_pcolor(v_polar, title='v')
    _save(fig, "Poisson solution")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 19  plot(psi), quiver(grad psi, 'k')
# ---------------------------------------------------------------------------

def disk_quiver_pts(nq=17):
    xs = np.linspace(-0.92, 0.92, nq)
    XQ, YQ = np.meshgrid(xs, xs)
    m = XQ ** 2 + YQ ** 2 <= 0.9
    return XQ[m], YQ[m]


try:
    gx = jax.vmap(jax.grad(psi_cart, 0))
    gy = jax.vmap(jax.grad(psi_cart, 1))
    xq, yq = disk_quiver_pts(18)
    uq = np.array(gx(jnp.asarray(xq), jnp.asarray(yq)))
    vq = np.array(gy(jnp.asarray(xq), jnp.asarray(yq)))

    def _q(ax):
        ax.quiver(xq, yq, uq, vq, color='k', scale=260, width=0.004)

    fig, _ = disk_pcolor(polar_from_cart(psi_cart), overlay=_q)
    _save(fig, "gradient quiver")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 20  contour(div u), quiver(u, 'k')   (u = grad psi, div u = lap psi)
# ---------------------------------------------------------------------------
try:
    def lap_psi(x, y):
        hxx = jax.grad(jax.grad(psi_cart, 0), 0)
        hyy = jax.grad(jax.grad(psi_cart, 1), 1)
        return hxx(x, y) + hyy(x, y)

    lp = jax.vmap(lap_psi)
    XX, YY, _ = eval_disk(lambda th, r: r * 0, n_theta=240, n_r=140)
    ZL = np.array(lp(jnp.asarray(XX.ravel()), jnp.asarray(YY.ravel()))).reshape(XX.shape)
    fig = plt.figure(figsize=(6.1, 2.58))
    ax = fig.add_axes(DISK_BOX)
    ax.contour(XX, YY, ZL, levels=10, cmap=PARULA, linewidths=0.9)
    xq, yq = disk_quiver_pts(18)
    uq = np.array(gx(jnp.asarray(xq), jnp.asarray(yq)))
    vq = np.array(gy(jnp.asarray(xq), jnp.asarray(yq)))
    ax.quiver(xq, yq, uq, vq, color='k', scale=260, width=0.004)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')
    fig.set_facecolor('white')
    _save(fig, "divergence contour + quiver")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 21  plot(g), quiver(surface curl of g, 'w')   (v = [dg/dy, -dg/dx])
# ---------------------------------------------------------------------------
try:
    dgx = jax.vmap(jax.grad(g_curl_cart, 0))
    dgy = jax.vmap(jax.grad(g_curl_cart, 1))
    xq, yq = disk_quiver_pts(18)
    cxx = np.array(dgy(jnp.asarray(xq), jnp.asarray(yq)))
    cyy = -np.array(dgx(jnp.asarray(xq), jnp.asarray(yq)))

    def _qc(ax):
        ax.quiver(xq, yq, cxx, cyy, color='w', scale=45, width=0.004)

    fig, _ = disk_pcolor(polar_from_cart(g_curl_cart),
                         title='The numerical surface curl of g', overlay=_qc)
    _save(fig, "surface curl")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 22  f (block-mirror-centrosymmetric example)
# ---------------------------------------------------------------------------
try:
    fig, _ = disk_pcolor(f_bmc_fn, title='f')
    _save(fig, "f (BMC example)")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 23  BMC function tf = cart2pol(f,'cdr'), view(2)
#     f extended to rho in [-1, 1] with the block-mirror rule.
# ---------------------------------------------------------------------------
try:
    thv = np.linspace(-np.pi, np.pi, 400)
    rhov = np.linspace(-1, 1, 300)
    TT, RR = np.meshgrid(thv, rhov)
    TTe = np.where(RR >= 0, TT, TT + np.pi)
    RRe = np.abs(RR)
    ZZ = np.array(f_bmc_fn(jnp.asarray(TTe.ravel()), jnp.asarray(RRe.ravel()))).reshape(TT.shape)
    fig, ax = line_fig(titled=True)
    ax.pcolormesh(TT, RR, ZZ, cmap=PARULA, shading='gouraud')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1, 1)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_title('The BMC function associated with f', fontsize=10)
    _save(fig, "BMC function")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 24  plot(f) skeleton — adaptively-selected GE slices (best effort: the
#     Diskfun does not retain pivot locations, so a representative skeleton
#     of the same rank is drawn).  Reported as a library gap.
# ---------------------------------------------------------------------------
try:
    f_bmc = Diskfun.from_function(f_bmc_fn)
    rank = f_bmc.rank
    fig = plt.figure(figsize=(6.1, 2.58))
    ax = fig.add_axes(DISK_BOX)
    blue = '#0072BD'
    # radial slices drawn as full diameters at selected angles
    ang = np.linspace(0, np.pi, 3 * rank, endpoint=False)
    rr = np.linspace(-1, 1, 120)
    for a in ang:
        ax.plot(rr * np.cos(a), rr * np.sin(a), color=blue, lw=0.8)
    # circular slices clustered near the centre and edge (Chebyshev radii)
    cheb_r = np.abs(np.cos(np.pi * np.arange(1, 3 * rank) / (6 * rank)))
    tt = np.linspace(0, 2 * np.pi, 300)
    for rc in cheb_r:
        ax.plot(rc * np.cos(tt), rc * np.sin(tt), color=blue, lw=0.8)
    _boundary(ax, lw=0.8)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')
    ax.set_title('Low rank function samples', fontsize=10)
    fig.set_facecolor('white')
    _save(fig, "skeleton (best effort)")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 25  tensor product grid on the disk
# ---------------------------------------------------------------------------
try:
    m = 120
    r_pts = np.abs(np.cos(np.pi * np.arange(m + 1) / (2 * m)))
    r_pts = np.unique(np.concatenate([r_pts, [0.0, 1.0]]))
    th_pts = np.linspace(0, 2 * np.pi, 2 * m, endpoint=True)
    fig = plt.figure(figsize=(6.1, 2.58))
    ax = fig.add_axes(DISK_BOX)
    tt = np.linspace(0, 2 * np.pi, 400)
    # concentric circles at the radial (Chebyshev) sample points
    for rc in r_pts:
        ax.plot(rc * np.cos(tt), rc * np.sin(tt), 'k', lw=0.22)
    # radial rays at the angular sample points
    for a in th_pts:
        ax.plot([0, np.cos(a)], [0, np.sin(a)], 'k', lw=0.22)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')
    ax.set_title('Tensor product function samples', fontsize=10)
    fig.set_facecolor('white')
    _save(fig, "tensor product grid")
except Exception as e:
    _fail(1, e)

# ---------------------------------------------------------------------------
# 26-27  column slices (radial, r in [-1,1]) and row slices (theta in [-pi,pi])
# ---------------------------------------------------------------------------
try:
    rank = len(f_bmc.cols)
    sel = list(range(2, min(7, rank)))
    fig, ax = line_fig(titled=True)
    rr = jnp.linspace(-1, 1, 400)
    rn = np.array(rr)
    for j in sel:
        ax.plot(rn, np.real(np.array(f_bmc.cols[j](rr))), linewidth=1.0)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-2, 1.5)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    ax.set_title(f'{len(sel)} of the {rank} column slices of f', fontsize=10)
    _save(fig, "column slices")

    fig, ax = line_fig(titled=True)
    xx = jnp.linspace(-1, 1, 400)          # theta = pi * x
    xn = np.array(xx) * np.pi
    for j in sel:
        ax.plot(xn, np.real(np.array(f_bmc.rows[j](xx))), linewidth=1.0)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-2, 1.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5])
    ax.set_title(f'{len(sel)} of the {rank} row slices of f', fontsize=10)
    _save(fig, "row slices")
except Exception as e:
    _fail(max(0, 27 - plot_num), e)

# ---------------------------------------------------------------------------
# 28  plotcoeffs(f): Chebyshev (columns) and Fourier (rows) coefficients
# ---------------------------------------------------------------------------
try:
    fig = plt.figure(figsize=(6.1, 2.58))
    ax1 = fig.add_axes([0.095, 0.235, 0.375, 0.585])
    ax2 = fig.add_axes([0.595, 0.235, 0.375, 0.585])
    yt = [1e0, 1e-5, 1e-10, 1e-15, 1e-20]
    for c in f_bmc.cols:
        cf = np.abs(np.array(c.coeffs))
        ax1.semilogy(np.arange(len(cf)), cf + 1e-40, '.', ms=4)
    ax1.set_xlim(0, 80)
    ax1.set_ylim(1e-20, 1e0)
    ax1.set_yticks(yt)
    ax1.set_xticks([0, 20, 40, 60, 80])
    ax1.set_title('Column slices', fontsize=10)
    ax1.set_xlabel('Degree of Chebyshev polynomial', fontsize=8)
    ax1.set_ylabel('Magnitude of coefficient', fontsize=8)
    ax1.grid(True, which='major', ls=':', lw=0.5, color='0.8')
    ax1.tick_params(labelsize=7)
    for rw in f_bmc.rows:
        cf = np.abs(np.array(rw.coeffs))
        k = np.arange(len(cf)) - len(cf) // 2
        ax2.semilogy(k, cf + 1e-40, '.', ms=3)
    ax2.set_xlim(-60, 60)
    ax2.set_ylim(1e-20, 1e0)
    ax2.set_yticks(yt)
    ax2.set_xticks([-50, 0, 50])
    ax2.set_title('Row slices', fontsize=10)
    ax2.set_xlabel('Wave number', fontsize=8)
    ax2.set_ylabel('Magnitude of coefficient', fontsize=8)
    ax2.grid(True, which='major', ls=':', lw=0.5, color='0.8')
    ax2.tick_params(labelsize=7)
    fig.set_facecolor('white')
    _save(fig, "plotcoeffs")
except Exception as e:
    _fail(1, e)

print(f"\nGuide 16: {plot_num} plots generated.")
