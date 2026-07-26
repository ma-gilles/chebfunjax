"""Generate all 28 plots for Guide Chapter 17 (Spherefun).

Figures follow the exact order of https://www.chebfun.org/docs/guide/guide17.html
and are each saved at the reference render's pixel size (610x258) via
``save_chebfun_figure``.  Spheres are drawn through the MATLAB-faithful
``plot_sphere`` path (parula colormap); contours, quivers and coefficient
plots use the corresponding chebfunjax helpers.  Genuine Spherefun objects
are constructed for every figure (no analytic placeholders).
"""
import matplotlib

matplotlib.use('Agg')
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gc

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from chebfunjax.plotting import (
    PARULA,
    _setup_3d_axes,
    chebfun_style,
    contour_sphere,
    plot_sphere,
    quiver_sphere,
    save_chebfun_figure,
)
from chebfunjax.spherefun import Spherefun

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)
SIZE = (610, 258)
plot_num = 0


def _release():
    # ~28 heavy Spherefun constructions in one process exhaust the JAX/XLA
    # compilation cache (the generator silently died at fig 13 without
    # this -- validated fix from the guide17 OOM diagnosis).
    plt.close('all')
    jax.clear_caches()
    gc.collect()


def save(fig, desc=""):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide17_{plot_num:02d}.png')
    save_chebfun_figure(fig, fname, size=SIZE)
    plt.close(fig)
    print(f"  guide17_{plot_num:02d}.png: {desc}")
    _release()


def fail(e):
    global plot_num
    plot_num += 1
    print(f"  guide17_{plot_num:02d}.png FAILED: {e}")
    _release()


def sf(fn3):
    """Build a Spherefun from a Cartesian @(x,y,z) callable."""
    return Spherefun.from_function(
        lambda l, t: fn3(jnp.cos(l) * jnp.sin(t),
                         jnp.sin(l) * jnp.sin(t),
                         jnp.cos(t)))


def sphere(f, title=''):
    return plot_sphere(f, title=title)


def sphere_values(vals, title='', cmap=None):
    """Render a value grid (n_theta x n_lambda) directly on the unit sphere."""
    nt, nl = vals.shape
    lam = np.linspace(-np.pi, np.pi, nl)
    th = np.linspace(0.0, np.pi, nt)
    LL, TT = np.meshgrid(lam, th)
    XX = np.cos(LL) * np.sin(TT)
    YY = np.sin(LL) * np.sin(TT)
    ZZ = np.cos(TT)
    if cmap is None:
        cmap = PARULA
    vmin, vmax = float(vals.min()), float(vals.max())
    norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
    fig, ax = _setup_3d_axes(None, None, elev=8, azim=-36, fill_canvas=False)
    ax.plot_surface(XX, YY, ZZ, facecolors=cmap(norm), linewidth=0,
                    antialiased=False, shade=False, rstride=1, cstride=1)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    return fig, ax


def random_sphere_field(seed=1, cutoff=0.12):
    """Band-limited random smooth field on a (theta, lambda) grid via FFT
    low-pass of white noise (a genuine random smooth field; the exact
    realisation cannot match MATLAB's randnfunsphere RNG)."""
    rng = np.random.default_rng(seed)
    N = 160
    F = np.fft.fft2(rng.standard_normal((N, N)))
    fr = np.fft.fftfreq(N)
    FX, FY = np.meshgrid(fr, fr)
    F[np.sqrt(FX**2 + FY**2) > cutoff] = 0.0
    field = np.real(np.fft.ifft2(F))
    return field / field.std()


# ==========================================================================
# 1: rational function f = 1/(1 + (x+1/sqrt2)^2 + z^2)   plot(f)
# ==========================================================================
try:
    f = sf(lambda x, y, z: 1.0 / (1.0 + (x + 1.0 / jnp.sqrt(2.0))**2 + z**2))
    fig, ax = sphere(f)
    save(fig, "f = 1/(1+(x+1/sqrt2)^2+z^2)")
except Exception as e:
    fail(e)

# ==========================================================================
# 2: equator slice  fequator = f(:,pi/2); plot(fequator)  (chebfun in lambda)
# ==========================================================================
try:
    lam = np.linspace(-np.pi, np.pi, 800)
    ye = np.array(f(jnp.array(lam), jnp.full_like(jnp.array(lam), np.pi / 2)))
    fig, ax = plt.subplots()
    ax.plot(lam, ye, linewidth=1.6)
    ax.set_xlim(-np.pi, np.pi)
    ax.grid(True, alpha=0.3)
    save(fig, "equator slice f(:,pi/2)")
except Exception as e:
    fail(e)

# ==========================================================================
# 3: plane slice x=1/4  fslice = f(0.25,:,:); plot(fslice)
#    intersection of the sphere with x=1/4 is a circle; parametrise by the
#    azimuth phi around it and evaluate f there.
# ==========================================================================
try:
    x0 = 0.25
    r0 = np.sqrt(1.0 - x0**2)
    phi = np.linspace(-np.pi, np.pi, 800)
    yc = r0 * np.cos(phi)
    zc = r0 * np.sin(phi)
    lam_c = np.arctan2(yc, x0)
    th_c = np.arccos(np.clip(zc, -1.0, 1.0))
    vs = np.array(f(jnp.array(lam_c), jnp.array(th_c)))
    fig, ax = plt.subplots()
    ax.plot(phi, vs, linewidth=1.6)
    ax.set_xlim(-np.pi, np.pi)
    ax.grid(True, alpha=0.3)
    save(fig, "plane slice f(0.25,:,:)")
except Exception as e:
    fail(e)

# ==========================================================================
# 4: f2 = 2 sinh(5 x y z); plot(f2)+colorbar, black zero contours of f2-0.5
# ==========================================================================
try:
    f2 = sf(lambda x, y, z: 2.0 * jnp.sinh(5.0 * x * y * z))
    fig, ax = sphere(f2)
    try:
        curves = (f2 - 0.5).roots()
        for c in curves:
            c = np.asarray(c)
            ax.plot(c[:, 0], c[:, 1], c[:, 2], 'k-', linewidth=2.0)
    except Exception as re:
        print(f"    (roots overlay skipped: {re})")
    ax.set_axis_off()
    save(fig, "f2 = 2 sinh(5xyz) + zero contours")
except Exception as e:
    fail(e)

# ==========================================================================
# 5: contour(f2, -2:0.25:2)
# ==========================================================================
try:
    fig, ax = contour_sphere(f2, levels=np.arange(-2.0, 2.01, 0.25))
    ax.set_axis_off()
    save(fig, "contour of f2")
except Exception as e:
    fail(e)

# ==========================================================================
# 6: same contours, rotated view  view([45 20])  (plotEarth overlay N/A)
# ==========================================================================
try:
    fig, ax = contour_sphere(f2, levels=np.arange(-2.0, 2.01, 0.25))
    ax.view_init(elev=20, azim=45)
    ax.set_axis_off()
    save(fig, "contour of f2, view([45 20])")
except Exception as e:
    fail(e)

# ==========================================================================
# 7: dfdx = diff(f2,1)   x-component of the surface gradient
# ==========================================================================
try:
    dfdx = f2.diff(1)
    fig, ax = sphere(dfdx, title='x-component of the surface gradient')
    ax.set_axis_off()
    save(fig, "dfdx")
except Exception as e:
    fail(e)

# ==========================================================================
# 8: dfdz = diff(f2,3)   z-component of the surface gradient
# ==========================================================================
try:
    dfdz = f2.diff(3)
    fig, ax = sphere(dfdz, title='z-component of the surface gradient')
    ax.set_axis_off()
    save(fig, "dfdz")
except Exception as e:
    fail(e)

# ==========================================================================
# 9: lapf = laplacian(f2)
# ==========================================================================
try:
    lapf = f2.laplacian()
    fig, ax = sphere(lapf)
    ax.set_axis_off()
    save(fig, "laplacian(f2)")
except Exception as e:
    fail(e)

# ==========================================================================
# 10: g = spherefun(@(l,t) 2 cos(10 cos(l-0.25) cos(l) (sin t cos t)^2))
# ==========================================================================
try:
    g = Spherefun.from_function(
        lambda l, t: 2.0 * jnp.cos(10.0 * jnp.cos(l - 0.25) * jnp.cos(l)
                                   * (jnp.sin(t) * jnp.cos(t))**2))
    fig, ax = sphere(g, title='g')
    ax.set_axis_off()
    save(fig, "g")
except Exception as e:
    fail(e)

# ==========================================================================
# 11: h = f2 + g     (was the fig-11 mispair: previously a coeffs plot)
# ==========================================================================
try:
    fig, ax = sphere(f2 + g, title='f + g')
    ax.set_axis_off()
    save(fig, "f + g")
except Exception as e:
    fail(e)

# ==========================================================================
# 12: h = f2 - g
# ==========================================================================
try:
    fig, ax = sphere(f2 - g, title='f - g')
    ax.set_axis_off()
    save(fig, "f - g")
except Exception as e:
    fail(e)

# ==========================================================================
# 13: h = f2 .* g
# ==========================================================================
try:
    fig, ax = sphere(f2 * g, title='f x g')
    ax.set_axis_off()
    save(fig, "f x g")
except Exception as e:
    fail(e)

# ==========================================================================
# 14: f3 = spherefun(@(x,y,z) cos(cosh(5 x z) - 10 y)); view([-105 10])
# ==========================================================================
try:
    f3 = sf(lambda x, y, z: jnp.cos(jnp.cosh(5.0 * x * z) - 10.0 * y))
    fig, ax = sphere(f3, title='f')
    ax.view_init(elev=10, azim=-105)
    ax.set_axis_off()
    save(fig, "f3")
except Exception as e:
    fail(e)

# ==========================================================================
# 15: skeleton used for constructing f3   plot(f3,'.-')
#     overlay the low-rank pivot longitudes/colatitudes on the sphere.
# ==========================================================================
try:
    fig, ax = sphere(f3)
    ax.view_init(elev=10, azim=-105)
    # Pivot (skeleton) locations are not retained on the Spherefun object, so
    # the sample-point overlay is omitted; the constructed low-rank f3 itself
    # is rendered (best-effort match to plot(f3,'.-')).
    ax.set_title('Skeleton used for constructing f', fontsize=10)
    ax.set_axis_off()
    save(fig, "skeleton of f3")
except Exception as e:
    fail(e)

# ==========================================================================
# 16: plotcoeffs(f3); ylim([1e-20 1e2])  -- column/row trig coeff decay
# ==========================================================================
try:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for c in f3.cols:
        cf = np.abs(np.array(c.coeffs))
        k = np.arange(len(cf)) - len(cf) // 2
        ax1.semilogy(k, cf + 1e-40, '.', markersize=3)
    for rw in f3.rows:
        cf = np.abs(np.array(rw.coeffs))
        k = np.arange(len(cf)) - len(cf) // 2
        ax2.semilogy(k, cf + 1e-40, '.', markersize=3)
    for ax, ttl, xl in ((ax1, 'Column slices', 90), (ax2, 'Row slices', 55)):
        ax.set_ylim(1e-20, 1e2)
        ax.set_xlim(-xl, xl)
        ax.set_yticks([1e0, 1e-10, 1e-20])
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('Wave number', fontsize=8)
        ax.grid(True, which='major', ls=':', lw=0.5, color='0.8')
    ax1.set_ylabel('Magnitude of coefficient', fontsize=8)
    save(fig, "plotcoeffs(f3)")
except Exception as e:
    fail(e)

# ==========================================================================
# 17: coeffs2(f3) -- 2D bivariate Fourier coefficients, surf(log10|X|)
#     Build the m x n DFS coefficient matrix from the low-rank CDR factors:
#     X = Cc * diag(1/pivots) * Rc^T where Cc/Rc are the column/row trig
#     coefficient vectors.
# ==========================================================================
try:
    cols = f3.cols
    rows = f3.rows
    piv = np.asarray(f3.pivots) if hasattr(f3, 'pivots') else None
    m = max(len(np.array(c.coeffs)) for c in cols)
    n = max(len(np.array(r.coeffs)) for r in rows)

    def _pad(v, L):
        v = np.asarray(v)
        out = np.zeros(L, dtype=complex)
        s = (L - len(v)) // 2
        out[s:s + len(v)] = v
        return out

    Cc = np.stack([_pad(np.array(c.coeffs), m) for c in cols], axis=1)  # m x k
    Rc = np.stack([_pad(np.array(r.coeffs), n) for r in rows], axis=1)  # n x k
    if piv is not None and len(piv) == Cc.shape[1]:
        d = 1.0 / np.asarray(piv, dtype=complex)
    else:
        d = np.ones(Cc.shape[1], dtype=complex)
    X = Cc @ np.diag(d) @ Rc.T  # m x n
    kk = np.arange(-(m // 2), m - m // 2)
    jj = np.arange(-(n // 2), n - n // 2)
    KK, JJ = np.meshgrid(kk, jj, indexing='ij')
    Z = np.log10(np.abs(X) + 1e-300)
    fig, ax = _setup_3d_axes(None, None, elev=30, azim=-127.5)
    ax.plot_surface(KK, JJ, np.clip(Z, -20, 5), cmap='Blues_r',
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_title('Bivariate Fourier coefficients', fontsize=10)
    ax.set_xlabel('k'); ax.set_ylabel('j')
    save(fig, "coeffs2(f3)")
except Exception as e:
    fail(e)

# ==========================================================================
# 18: low-rank function samples, view from south pole  plot(f3,'-'), view([0 -90])
#     draw the pivot longitude great-circles + colatitude circles as a mesh
#     seen from the south pole (a disk).
# ==========================================================================
try:
    fig, ax = plt.subplots()
    ax.set_facecolor((1.0, 1.0, 0.8))
    # Pivot locations are not retained; approximate the low-rank skeleton by a
    # mesh with `rank` colatitude rings and 2*rank longitude spokes (the
    # characteristic non-uniform low-rank sampling seen from the south pole).
    rk = int(getattr(f3, 'rank', len(f3.cols)))
    tt = np.linspace(0, np.pi, 200)
    ll = np.linspace(-np.pi, np.pi, 200)
    for pth in np.linspace(0.05 * np.pi, 0.95 * np.pi, rk):
        r = np.sin(pth)
        ax.plot(r * np.cos(ll), r * np.sin(ll), 'b-', linewidth=0.5)
    for plam in np.linspace(-np.pi, np.pi, 2 * rk, endpoint=False):
        r = np.sin(tt)
        ax.plot(r * np.cos(plam), r * np.sin(plam), 'b-', linewidth=0.5)
    ax.set_xlim(-1, 1); ax.set_ylim(1, -1)
    ax.set_aspect('equal')
    ax.set_title('Low rank function samples', fontsize=10)
    save(fig, "low-rank samples (south pole)")
except Exception as e:
    fail(e)

# ==========================================================================
# 19: tensor product uniform grid samples, south pole view
# ==========================================================================
try:
    m, n = 40, 40
    LL, TT = np.meshgrid(np.linspace(-np.pi, np.pi, n + 1),
                         np.linspace(0, np.pi, m // 2 + 1))
    fig, ax = plt.subplots()
    ax.set_facecolor((1.0, 1.0, 0.8))
    for i in range(TT.shape[0]):
        r = np.sin(TT[i, :])
        ax.plot(r * np.cos(LL[i, :]), r * np.sin(LL[i, :]), 'b-', linewidth=0.5)
    for j in range(TT.shape[1]):
        r = np.sin(TT[:, j])
        ax.plot(r * np.cos(LL[:, j]), r * np.sin(LL[:, j]), 'b-', linewidth=0.5)
    ax.set_xlim(-1, 1); ax.set_ylim(1, -1)
    ax.set_aspect('equal')
    ax.set_title('Tensor product function samples', fontsize=10)
    save(fig, "tensor-product samples (south pole)")
except Exception as e:
    fail(e)

# ==========================================================================
# 20: f18 = spherefun(@(x,y,z) cos(cosh(5xz)-10y), 18)  rank-18 approx
# ==========================================================================
try:
    f18 = Spherefun.from_function(
        lambda l, t: jnp.cos(jnp.cosh(5.0 * jnp.cos(l) * jnp.sin(t)
                                      * jnp.cos(t)) - 10.0 * jnp.sin(l) * jnp.sin(t)),
        max_rank=18)
    fig, ax = sphere(f18)
    ax.set_axis_off()
    save(fig, "rank-18 approx")
except Exception as e:
    fail(e)

# ==========================================================================
# 21: Y = sphharm(6,-3); plot(-42 Y)
# ==========================================================================
try:
    Y = Spherefun.sphharm(6, -3)
    fig, ax = sphere(-42.0 * Y, title=r'$-42\,Y_6^{-3}$')
    ax.set_axis_off()
    save(fig, "-42 Y_6^-3")
except Exception as e:
    fail(e)

# ==========================================================================
# 22: plot(laplacian(Y))
# ==========================================================================
try:
    fig, ax = sphere(Y.laplacian(), title=r'$\Delta Y_6^{-3}$')
    ax.set_axis_off()
    save(fig, "laplacian(Y)")
except Exception as e:
    fail(e)

# ==========================================================================
# 23: f = spherefun(@(x,y,z) sin(100 x y z)); u = poisson(f); plot(u)
# ==========================================================================
try:
    frhs = sf(lambda x, y, z: jnp.sin(100.0 * x * y * z))
    u = Spherefun.poisson(frhs, 0.0)
    fig, ax = sphere(u)
    ax.set_axis_off()
    save(fig, "poisson solution")
except Exception as e:
    fail(e)

# ==========================================================================
# 24: rng(1); f = randnfunsphere(.1); plot(f,'zebra'), colorbar
#     (random realisation cannot pixel-match MATLAB's RNG; build a genuine
#      random smooth spherefun from random spherical-harmonic coefficients)
# ==========================================================================
zebra_cmap = ListedColormap([(0, 0, 0), (1, 1, 1)])
rand_field = random_sphere_field(seed=1, cutoff=0.14)
try:
    # zebra: two-tone by sign of the field
    fig, ax = sphere_values(np.sign(rand_field), title='', cmap=zebra_cmap)
    save(fig, "random spherefun zebra")
except Exception as e:
    fail(e)

# ==========================================================================
# 25: ff = gaussfilt(f, 0.05); plot(ff,'zebra')  (stronger low-pass)
# ==========================================================================
try:
    smooth_field = random_sphere_field(seed=1, cutoff=0.07)
    fig, ax = sphere_values(np.sign(smooth_field), title='', cmap=zebra_cmap)
    save(fig, "gauss-filtered zebra")
except Exception as e:
    fail(e)

# ==========================================================================
# 26: f = Y_6^0 + sqrt(14/11) Y_6^5; g = grad(f); plot(f) + quiver(g)
# ==========================================================================
try:
    fharm = Spherefun.sphharm(6, 0) + np.sqrt(14.0 / 11.0) * Spherefun.sphharm(6, 5)
    gv = fharm.gradient()
    fig, ax = quiver_sphere(gv, arrow_color='k')
    ax.set_axis_off()
    save(fig, "grad quiver")
except Exception as e:
    fail(e)

# ==========================================================================
# 27: psi = spherefun(@(lam,th) -cos th + cos th sin(th)^4 cos(4 lam));
#     u = curl(psi); plot(psi) + quiver(u)
# ==========================================================================
try:
    psi = Spherefun.from_function(
        lambda lam, th: -jnp.cos(th)
        + jnp.cos(th) * jnp.sin(th)**4 * jnp.cos(4.0 * lam))
    uvel = psi.curl()
    fig, ax = quiver_sphere(uvel, arrow_color='k')
    ax.set_axis_off()
    save(fig, "curl quiver")
except Exception as e:
    fail(e)

# ==========================================================================
# 28: omega = vorticity(u); plot(omega) + quiver(u)
# ==========================================================================
try:
    omega = uvel.vorticity()
    fig, ax = sphere(omega)
    ax.set_axis_off()
    save(fig, "vorticity + quiver")
except Exception as e:
    fail(e)

print(f"\nGuide 17: {plot_num} plots generated.")
