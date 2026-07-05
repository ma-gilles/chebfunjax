"""Generate all plots for Guide Chapter 19 (SPIN/SPIN2/SPIN3/SPINSPHERE).

Uses PARULA colormap and MATLAB-style sphere plots for spinsphere.
"""
import matplotlib

matplotlib.use('Agg')
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    PARULA,
    _setup_3d_axes,
    chebfun_style,
)

chebfun_style()

OUT = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUT, exist_ok=True)
plot_num = 0

def save(fig, desc=""):
    global plot_num
    plot_num += 1
    fname = os.path.join(OUT, f'guide19_{plot_num:02d}.png')
    from chebfunjax.plotting import save_chebfun_figure
    save_chebfun_figure(fig, fname, size=(600, 270))
    plt.close(fig)
    print(f"  guide19_{plot_num:02d}.png: {desc}")

def sphere_from_latlon(ll, tt, u_vals, title='', cmap=None):
    """Render lat/lon data on a 3D sphere with PARULA."""
    if cmap is None:
        cmap = PARULA
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Convert (lambda, theta) -> Cartesian on the unit sphere
    # theta here is colatitude in [0, pi] (or may be [-pi, pi] for spinsphere)
    # Handle the spinsphere convention where theta is in [-pi, pi]
    TH_phys = tt
    XX = np.sin(TH_phys) * np.cos(ll)
    YY = np.sin(TH_phys) * np.sin(ll)
    ZZ = np.cos(TH_phys)

    u_r = np.real(np.asarray(u_vals))
    fmin, fmax = float(u_r.min()), float(u_r.max())
    if fmax > fmin:
        norm_v = (u_r - fmin) / (fmax - fmin)
    else:
        norm_v = np.full_like(u_r, 0.5)
    fcolors = cmap_obj(norm_v)

    fig, ax = _setup_3d_axes(None, None, elev=8, azim=-36)
    ax.plot_surface(XX, YY, ZZ, facecolors=fcolors, linewidth=0,
                    antialiased=True, alpha=0.95, shade=False,
                    rstride=1, cstride=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    if title:
        ax.set_title(title, fontsize=10, pad=0)
    return fig, ax

from chebfunjax.spin import SpinOp, SpinOp2, spin, spin2

# Plot 01: KdV
try:
    x, t, u = spin('KdV', N=256, dt=1e-6)
    fig, ax = plt.subplots()
    ax.plot(x, np.real(u), color=CHEBFUN_BLUE, lw=1.8)
    ax.set_title(f'KdV at t = {t:.4g}', fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "KdV")
except Exception as e:
    plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plot 02: Allen-Cahn t=500
try:
    x, t, u = spin('AC', N=256, dt=0.1)
    fig, ax = plt.subplots()
    ax.plot(x, np.real(u), color=CHEBFUN_BLUE, lw=1.8)
    ax.set_title(f'Allen-Cahn at t = {t:.0f}', fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "AC")
except Exception as e:
    plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plot 03: AC t=100
try:
    op = SpinOp.from_name('AC'); op.tspan = (0., 100.)
    x, t, u = spin(op, N=256, dt=0.1)
    fig, ax = plt.subplots()
    ax.plot(x, np.real(u), color=CHEBFUN_BLUE, lw=1.8)
    ax.set_title(f'Allen-Cahn at t = {t:.0f}', fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "AC t=100")
except Exception as e:
    plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plot 04: AC custom IC
try:
    op = SpinOp.from_name('AC'); op.tspan = (0., 100.)
    op.u0 = lambda x: -1 + 4*jnp.exp(-19*(x - jnp.pi)**2)
    x, t, u = spin(op, N=256, dt=0.1)
    fig, ax = plt.subplots()
    ax.plot(x, np.real(u), color=CHEBFUN_BLUE, lw=1.8)
    ax.set_title(f'AC custom IC at t = {t:.0f}', fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "AC custom IC")
except Exception as e:
    plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plot 05: IC plot
try:
    x_ic = np.linspace(0, 2*np.pi, 256, endpoint=False)
    fig, ax = plt.subplots()
    ax.plot(x_ic, -1 + 4*np.exp(-19*(x_ic-np.pi)**2), color=CHEBFUN_BLUE, lw=1.8)
    ax.set_title('Initial condition', fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylim(-1.5, 3.5)
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "IC")
except Exception as e:
    plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plot 06: Waterfall placeholder (AC at t=30)
try:
    op = SpinOp.from_name('AC')
    op.u0 = lambda x: -1 + 4*jnp.exp(-19*(x - jnp.pi)**2)
    op.tspan = (0., 30.)
    x_w, t_w, u_w = spin(op, N=256, dt=0.1)
    fig, ax = plt.subplots()
    ax.plot(x_w, np.real(u_w), color=CHEBFUN_BLUE, lw=1.8)
    ax.set_title(f'AC at t = {t_w:.0f}', fontsize=10)
    ax.set_xlabel('x', fontsize=9)
    ax.grid(True, alpha=0.3, ls='--', lw=0.6)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig, "waterfall placeholder")
except Exception as e:
    plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plots 07-10: 2D GL at various times.
# MATLAB's demo IC is randnfun2(4, [0 100 0 100], 'trig') — a RANDOM
# band-limited field normalized to max 1, so the published pattern is
# one particular draw and cannot be reproduced exactly. We match its
# statistics: random Fourier modes up to wavelength 4, conjugate-
# symmetric (real field), fixed seed for reproducibility.
def _randnfun2_trig(N, dom_len=100.0, wavelength=4.0, seed=7):
    rng = np.random.default_rng(seed)
    kmax = int(dom_len / wavelength)
    c = np.zeros((N, N), dtype=complex)
    ks = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    for i, ki in enumerate(ks):
        for j, kj in enumerate(ks):
            if abs(ki) <= kmax and abs(kj) <= kmax:
                c[i, j] = rng.standard_normal() + 1j * rng.standard_normal()
    vals = np.real(np.fft.ifft2(c))
    return vals / np.max(np.abs(vals))


for tfin in [10, 20, 30, 100]:
    try:
        op = SpinOp2.from_name('GL'); op.tspan = (0., float(tfin))
        N_gl = 128
        u0_vals = _randnfun2_trig(N_gl)
        op.u0 = lambda x, y, _v=jnp.asarray(u0_vals): _v
        xx, yy, t, u = spin2(op, N=N_gl, dt=5e-2)
        u_r = np.real(np.asarray(u)) if not isinstance(u, list) else np.real(np.asarray(u[0]))
        fig, ax = plt.subplots()
        ax.pcolormesh(xx, yy, u_r, cmap=PARULA, shading='gouraud')
        ax.set_aspect('equal'); ax.axis('off')
        fig.set_facecolor('white'); fig.tight_layout()
        save(fig, f"GL t={tfin}")
    except Exception as e:
        plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plot 11: AC sphere IC -- render on 3D sphere
try:
    from chebfunjax.spin import SpinOpSphere
    op = SpinOpSphere.from_name('AC')
    N_sp = 64; lam = np.linspace(-np.pi, np.pi, N_sp, endpoint=False)
    theta = np.linspace(0, np.pi, N_sp)
    LL, TT = np.meshgrid(lam, theta)
    u0 = np.real(np.asarray(op.u0(LL, TT)))
    fig, ax = sphere_from_latlon(LL, TT, u0, title='AC sphere IC')
    save(fig, "AC sphere IC")
except Exception as e:
    plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plots 12-14: AC sphere at t=2,5,10
try:
    from chebfunjax.spin import spinsphere
    for tfin in [2, 5, 10]:
        op = SpinOpSphere.from_name('AC'); op.tspan = (0., float(tfin))
        grids, t, u = spinsphere(op, N=32, dt=5e-2)
        ll, tt = grids; u_r = np.real(np.asarray(u))
        # Try to render on 3D sphere
        try:
            fig, ax = sphere_from_latlon(ll, tt, u_r, title=f'AC sphere t = {tfin}')
        except Exception:
            # Fallback to flat
            fig, ax = plt.subplots()
            ax.pcolormesh(ll, tt, u_r, cmap=PARULA, shading='auto')
            ax.set_title(f'AC sphere t = {tfin}', fontsize=10)
            ax.set_xlabel('lambda', fontsize=9)
            ax.set_ylabel('theta', fontsize=9)
            fig.set_facecolor('white'); fig.tight_layout()
        save(fig, f"AC sphere t={tfin}")
except Exception as e:
    for _ in range(3):
        plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

# Plots 15-18: GL sphere
try:
    for tfin in [0.1, 10, 20, 30]:
        op = SpinOpSphere.from_name('GL'); op.tspan = (0., float(tfin))
        grids, t, u = spinsphere(op, N=32, dt=5e-2)
        ll, tt = grids; u_abs = np.abs(np.asarray(u))
        try:
            fig, ax = sphere_from_latlon(ll, tt, u_abs, title=f'GL sphere |u| t = {tfin:.1f}')
        except Exception:
            fig, ax = plt.subplots()
            ax.pcolormesh(ll, tt, u_abs, cmap=PARULA, shading='auto')
            ax.set_title(f'GL sphere |u| t = {tfin:.1f}', fontsize=10)
            ax.set_xlabel('lambda', fontsize=9)
            ax.set_ylabel('theta', fontsize=9)
            fig.set_facecolor('white'); fig.tight_layout()
        save(fig, f"GL sphere t={tfin}")
except Exception as e:
    for _ in range(4):
        plot_num += 1; print(f"  guide19_{plot_num:02d}.png FAILED: {e}")

print(f"\nGuide 19: {plot_num} plots generated.")
