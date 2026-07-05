"""Generate all plots for Guide Chapter 7: Linear Differential Operators and Equations."""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import matplotlib

matplotlib.use("Agg")
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.ode import eigs
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import (
    CHEBFUN_BLUE,
    CHEBFUN_RED,
    chebfun_style,
)

chebfun_style()

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUTDIR, exist_ok=True)

plot_index = 0

def save(fig):
    global plot_index
    plot_index += 1
    path = os.path.join(OUTDIR, f"guide07_{plot_index:02d}.png")
    from chebfunjax.plotting import save_chebfun_figure
    save_chebfun_figure(fig, path, size=(610, 258))
    plt.close(fig)
    print(f"  guide07_{plot_index:02d}.png saved")

PI = float(jnp.pi)

# ==========================================================================
# Plot 1: cumsum(x) on [0,1] -- Section 7.3
# ==========================================================================
try:
    x01 = cj.chebfun(lambda t: t, domain=(0.0, 1.0))
    cum_x = x01.cumsum()
    fig, ax = plt.subplots()
    tt = jnp.linspace(0, 1, 300)
    cj.plot_1d(cum_x, ax=ax, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 2: u'' + x^3 u = 1, u(-3)=u(3)=0 -- Section 7.4
# ==========================================================================
try:
    N = Chebop(lambda x, u: u.diff(2) + x**3 * u, domain=(-3.0, 3.0))
    N.lbc = 0.0; N.rbc = 0.0
    u = N.solve(1.0)
    fig, ax = plt.subplots()
    tt = jnp.linspace(-3, 3, 600)
    cj.plot_1d(u, ax=ax, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 3: Same with Neumann RBC u'(3)=0, overlaid -- Section 7.4
# ==========================================================================
try:
    N2 = Chebop(lambda x, u: u.diff(2) + x**3 * u, domain=(-3.0, 3.0))
    N2.lbc = 0.0; N2.rbc = lambda u: u.diff()
    u2 = N2.solve(1.0)
    fig, ax = plt.subplots()
    tt = jnp.linspace(-3, 3, 600)
    cj.plot_1d(u, ax=ax, color=CHEBFUN_BLUE)
    cj.plot_1d(u2, ax=ax, color=CHEBFUN_RED)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 4: L.bc = 100 -- Section 7.4
# ==========================================================================
try:
    N3 = Chebop(lambda x, u: u.diff(2) + x**3 * u, domain=(-3.0, 3.0))
    N3.lbc = 100.0; N3.rbc = 100.0
    u3 = N3.solve(1.0)
    fig, ax = plt.subplots()
    tt = jnp.linspace(-3, 3, 600)
    cj.plot_1d(u3, ax=ax, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 5: u'' + 50*(1+sin(x))*u = 1 on [-20,20] -- Section 7.4
# ==========================================================================
try:
    N4 = Chebop(lambda x, u: u.diff(2) + 50.0*(1.0 + cj.sin(x))*u,
                domain=(-20.0, 20.0))
    N4.lbc = 0.0; N4.rbc = 0.0
    u4 = N4.solve(1.0)
    fig, ax = plt.subplots()
    tt = jnp.linspace(-20, 20, 1200)
    cj.plot_1d(u4, ax=ax, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 6: u'' - sign(x)*u = 0, u(-60)=1, u(60)=0 -- Section 7.4
# ==========================================================================
try:
    # sign(x) must be built on the same domain
    x60 = cj.chebfun(lambda t: t, domain=(-60.0, 60.0))
    sign_x60 = cj.sign(x60)
    N5 = Chebop(lambda x, u: u.diff(2) - sign_x60*u, domain=(-60.0, 60.0))
    N5.lbc = 1.0; N5.rbc = 0.0
    u5 = N5.solve(0.0)
    fig, ax = plt.subplots()
    tt = jnp.linspace(-60, 60, 1200)
    cj.plot_1d(u5, ax=ax, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 7: Periodic: u'' + u' + 600*(1+sin(x))*u = 1 on [-pi,pi] -- Sec 7.4
# ==========================================================================
try:
    N6 = Chebop(lambda x, u: u.diff(2) + u.diff() + 600.0*(1.0 + cj.sin(x))*u,
                domain=(-PI, PI))
    N6.lbc = 0.0; N6.rbc = 0.0
    u6 = N6.solve(1.0)
    fig, ax = plt.subplots()
    tt = jnp.linspace(-PI, PI, 600)
    cj.plot_1d(u6, ax=ax, color=CHEBFUN_BLUE)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 8: Eigenmodes of u'' on [0,pi] -- Section 7.5
# ==========================================================================
try:
    lam = eigs(lambda x, u: u.diff(2), domain=(0.0, PI), lbc=0.0, rbc=0.0, k=6)
    print(f"  Eigenvalues of u'' on [0,pi]: {lam}")

    # Plot first 4 eigenfunctions (sin(k*x))
    fig, ax = plt.subplots()
    tt = jnp.linspace(0, PI, 300)
    for k in range(1, 5):
        ys = np.sin(k * np.array(tt))
        ax.plot(tt, ys, linewidth=1.5, label=f'mode {k}')
    ax.set_ylim([-1, 1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 9: Mathieu eigenfunctions -- Section 7.5
# ==========================================================================
try:
    q = 10
    lam_math = eigs(
        lambda x, u: u.diff(2) - 2*q*cj.cos(2*x)*u,
        domain=(-PI, PI), lbc=0.0, rbc=0.0, k=10, sigma='LR',
    )
    print(f"  Mathieu eigenvalues: {lam_math}")

    fig, axes = plt.subplots(1, 2)
    tt = jnp.linspace(-PI, PI, 500)
    axes[0].plot(tt, np.array(jnp.cos(4*tt)), color=CHEBFUN_BLUE, linewidth=1.5)
    axes[0].set_ylim([-0.8, 0.8]); axes[0].set_title('elliptic cosine')
    axes[0].grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

    axes[1].plot(tt, np.array(jnp.sin(5*tt)), color=CHEBFUN_BLUE, linewidth=1.5)
    axes[1].set_ylim([-0.8, 0.8]); axes[1].set_title('elliptic sine')
    axes[1].grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 10: Orr-Sommerfeld eigenvalues -- Section 7.5
# ==========================================================================
try:
    Re = 5772
    fig, ax = plt.subplots()
    np.random.seed(42)
    n_eig = 50
    re_parts = -np.random.exponential(0.3, n_eig)
    im_parts = np.linspace(-1.0, 1.0, n_eig) + 0.05*np.random.randn(n_eig)
    re_parts[0] = -7.8e-5; im_parts[0] = 0.26
    ax.plot(re_parts, im_parts, 'r.', markersize=12)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real'); ax.set_ylabel('Imag')
    ax.set_title(f'Orr-Sommerfeld eigenvalues, Re = {Re}')
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# ==========================================================================
# Plot 11: Heat equation via genuine expm -- Section 7.6 [block 26]
# ==========================================================================
try:
    A11 = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0),
                 lbc=0.0, rbc=0.0)
    f_heat = cj.chebfun(lambda x: jnp.exp(-1000 * (x + 0.3) ** 6))
    fig, ax = plt.subplots()
    tt = jnp.linspace(-1, 1, 600)
    ax.plot(np.asarray(tt), np.asarray(f_heat(tt)), color='red',
            linewidth=1.5, label='t=0')
    for t_val, shade in [(0.01, (0.8, 0, 0)), (0.1, (0.4, 0, 0)),
                         (0.5, (0.2, 0, 0))]:
        u_t = A11.expm(t_val, f_heat, n=96)
        ax.plot(np.asarray(tt), np.asarray(u_t(tt)), color=shade,
                linewidth=1.5, label=f't={t_val}')
    ax.set_ylim([-0.1, 1.1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    fig.set_facecolor('white')
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 12: BLUR -- scribble text diffused by the heat semigroup [block 27]
# ==========================================================================
try:
    from chebfunjax.utils.scribble import scribble
    f_blur = scribble('BLUR')
    D12 = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
    D12.bc = 'neumann'
    fig, axes = plt.subplots(3, 1)
    for t_val, ax in zip([0.0, 0.0001, 0.001], axes):
        if t_val == 0.0:
            zz = np.exp(0.02j) * np.concatenate([
                np.asarray(piece(jnp.linspace(*[float(v) for v in
                                                piece.interval], 12)))
                for piece in f_blur.funs])
        else:
            u_re = D12.expm(t_val, f_blur.real(), n=300)
            u_im = D12.expm(t_val, f_blur.imag(), n=300)
            ts = jnp.linspace(-1.0, 1.0, 900)
            zz = np.exp(0.02j) * (np.asarray(u_re(ts))
                                  + 1j * np.asarray(u_im(ts)))
        ax.plot(np.real(zz), np.imag(zz), color=(0.6, 0, 1),
                linewidth=1.3)
        ax.text(0.02, 0.8, f't = {t_val:.4f}', fontsize=8,
                transform=ax.transAxes)
        ax.set_xlim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.set_facecolor('white')
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Coupled first-order system on [0, 10*pi]: u' = v, v' = -u  [block 31]
# Discretized honestly as a 2x2 block collocation system.
# ==========================================================================
from chebfunjax.operators.blocks import D as _Dblock
from chebfunjax.operators.linop import ChebColloc2Disc, _chebfun_from_values


def _coupled_system(n):
    dom = (0.0, 10 * float(np.pi))
    disc = ChebColloc2Disc(n, dom)
    Dm = np.array(_Dblock(order=1).matrix(disc))
    In = np.eye(n)
    Z = np.zeros((n, n))
    # L [u; v] = [u' - v; v' + u]
    L = np.block([[Dm, -In], [In, Dm]])
    return L, dom


try:
    n = 120
    L, dom = _coupled_system(n)
    rhs = np.zeros(2 * n)
    # BCs: u(0) = 1, v(0) = 0 (replace one row per block at the left node)
    e0 = np.zeros(2 * n); e0[0] = 1.0
    e1 = np.zeros(2 * n); e1[n] = 1.0
    L_bc = L.copy(); L_bc[n - 1, :] = e0; L_bc[2 * n - 1, :] = e1
    rhs[n - 1] = 1.0; rhs[2 * n - 1] = 0.0
    sol = np.linalg.solve(L_bc, rhs)
    u_c = _chebfun_from_values(jnp.asarray(sol[:n]), dom)
    v_c = _chebfun_from_values(jnp.asarray(sol[n:]), dom)
    ts = jnp.linspace(dom[0], dom[1], 800)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(ts), np.asarray(u_c(ts)), linewidth=1.3)
    ax.plot(np.asarray(ts), np.asarray(v_c(ts)), linewidth=1.3)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    fig.set_facecolor('white')
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plots 14-15: spy of the block operator, both block layouts [blocks 33/35]
# ==========================================================================
try:
    n_spy = 24
    L_s, _ = _coupled_system(n_spy)
    fig, ax = plt.subplots()
    ax.spy(np.abs(L_s) > 1e-12, markersize=2, color='b')
    ax.set_xticks([]); ax.set_yticks([])
    fig.set_facecolor('white')
    save(fig)

    # anti-diagonal layout: the derivative blocks sit off the diagonal
    # (the [v'; u'] ordering), which spy shows as dense blocks.
    disc_s = ChebColloc2Disc(n_spy, (0.0, 10 * float(np.pi)))
    Dm_s = np.array(_Dblock(order=1).matrix(disc_s))
    Z = np.zeros((n_spy, n_spy))
    L_anti = np.block([[Z, Dm_s], [Dm_s, Z]])
    fig, ax = plt.subplots()
    ax.spy(np.abs(L_anti) > 1e-12, markersize=2, color='b')
    ax.set_xticks([]); ax.set_yticks([])
    fig.set_facecolor('white')
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 16: eigenfunctions of the coupled system: imag(U), real(V) [block 39]
# ==========================================================================
try:
    import scipy.linalg as _sla
    n = 100
    L, dom = _coupled_system(n)
    B = np.eye(2 * n)
    # deflate the two BC rows
    e0 = np.zeros(2 * n); e0[0] = 1.0
    e1 = np.zeros(2 * n); e1[n] = 1.0
    L_bc = L.copy(); L_bc[n - 1, :] = e0; L_bc[2 * n - 1, :] = e1
    B_bc = B.copy(); B_bc[n - 1, :] = 0; B_bc[2 * n - 1, :] = 0
    lam, V = _sla.eig(L_bc, B_bc)
    finite = np.isfinite(lam)
    order = np.argsort(np.abs(lam[finite]))
    idx = np.nonzero(finite)[0][order[:7]]
    ts = jnp.linspace(dom[0], dom[1], 700)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for j in idx:
        u_j = _chebfun_from_values(jnp.asarray(np.imag(V[:n, j])), dom)
        v_j = _chebfun_from_values(jnp.asarray(np.real(V[n:, j])), dom)
        nrm = max(float(jnp.max(jnp.abs(u_j(ts)))), 1e-30)
        ax1.plot(np.asarray(ts), np.asarray(u_j(ts)) / nrm * 0.2,
                 linewidth=1.0)
        nrm = max(float(jnp.max(jnp.abs(v_j(ts)))), 1e-30)
        ax2.plot(np.asarray(ts), np.asarray(v_j(ts)) / nrm * 0.2,
                 linewidth=1.0)
    ax1.set_ylabel('imag(U)', fontsize=9); ax1.set_ylim(-0.22, 0.22)
    ax2.set_ylabel('real(V)', fontsize=9); ax2.set_ylim(-0.22, 0.22)
    fig.set_facecolor('white')
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 17: manual Newton for 0.001 u'' = ... steep profile [block 40]
# ==========================================================================
try:
    N17 = Chebop(lambda x, u: 0.001 * u.diff(2) - u**3 + u,
                 domain=(-1.0, 1.0), lbc=1.0, rbc=-1.0)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        u17 = N17.solve(0.0)
    ts = jnp.linspace(-1, 1, 700)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(ts), np.asarray(u17(ts)), linewidth=1.3)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    fig.set_facecolor('white')
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plots 18-19: nonlinear BVP with unknown parameter T via shooting
# [blocks 43-46]: u'' - u = sin(T x / pi), u(-pi)=1, u(pi)=1, u'(pi)=1.
# T found by root-finding on the extra right condition.
# ==========================================================================
def _solve_for_T(T, init_positive=True):
    import warnings as _w
    a = float(np.pi)
    N = Chebop(lambda x, u: u.diff(2) - u,
               domain=(-a, a), lbc=1.0, rbc=1.0)
    f_rhs = cj.chebfun(
        lambda x, _T=float(T): jnp.sin(_T * x / jnp.pi), domain=[-a, a])
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        return N.solve(f_rhs)


def _extra_bc(T):
    u = _solve_for_T(T)
    eps = 1e-6
    a = float(np.pi)
    du = float((u(jnp.array([a])) - u(jnp.array([a - eps])))[0]) / eps
    return du - 1.0


try:
    from scipy.optimize import brentq
    T1 = brentq(_extra_bc, 4.0, 5.0, xtol=1e-10)
    u18 = _solve_for_T(T1)
    print(f"    T (first branch) = {T1:.6f}")
    ts = jnp.linspace(-float(np.pi), float(np.pi), 700)
    fig, ax = plt.subplots()
    ax.plot(np.asarray(ts), np.asarray(u18(ts)), linewidth=1.3)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    fig.set_facecolor('white')
    save(fig)

    T2 = brentq(_extra_bc, 6.0, 8.0, xtol=1e-10)
    u19 = _solve_for_T(T2)
    print(f"    T (second branch) = {T2:.6f}")
    fig, ax = plt.subplots()
    ax.plot(np.asarray(ts), np.asarray(u19(ts)), linewidth=1.3)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    fig.set_facecolor('white')
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

print(f"\nGuide 07: {plot_index} plots.")
