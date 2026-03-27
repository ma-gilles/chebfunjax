"""Generate all plots for Guide Chapter 7: Linear Differential Operators and Equations."""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax.numpy as jnp
import numpy as np
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, CHEBFUN_BLUE, CHEBFUN_RED, CHEBFUN_GREEN, CHEBFUN_ORANGE
from chebfunjax.operators.chebop import Chebop
from chebfunjax.chebfun1d.ode import bvp, eigs

chebfun_style()

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUTDIR, exist_ok=True)

plot_index = 0

def save(fig):
    global plot_index
    plot_index += 1
    path = os.path.join(OUTDIR, f"guide07_{plot_index:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide07_{plot_index:02d}.png saved")

PI = float(jnp.pi)

# ==========================================================================
# Plot 1: cumsum(x) on [0,1] -- Section 7.3
# ==========================================================================
try:
    x01 = cj.chebfun(lambda t: t, domain=(0.0, 1.0))
    cum_x = x01.cumsum()
    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(0, 1, 300)
    ax.plot(tt, np.array(cum_x(tt)), color=CHEBFUN_BLUE, linewidth=1.8)
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
    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-3, 3, 600)
    ax.plot(tt, np.array(u(tt)), color=CHEBFUN_BLUE, linewidth=1.8)
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
    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-3, 3, 600)
    ax.plot(tt, np.array(u(tt)), color=CHEBFUN_BLUE, linewidth=1.5)
    ax.plot(tt, np.array(u2(tt)), color=CHEBFUN_RED, linewidth=1.5)
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
    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-3, 3, 600)
    ax.plot(tt, np.array(u3(tt)), color=CHEBFUN_BLUE, linewidth=1.8)
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
    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-20, 20, 1200)
    ax.plot(tt, np.array(u4(tt)), color=CHEBFUN_BLUE, linewidth=0.8)
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
    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-60, 60, 1200)
    ax.plot(tt, np.array(u5(tt)), color=CHEBFUN_BLUE, linewidth=1.0)
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
    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-PI, PI, 600)
    ax.plot(tt, np.array(u6(tt)), color=CHEBFUN_BLUE, linewidth=1.0)
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
    fig, ax = plt.subplots(figsize=(6, 4))
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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
    fig, ax = plt.subplots(figsize=(6, 5))
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
# Plot 11: Heat equation expm -- Section 7.6
# ==========================================================================
try:
    f_heat = cj.chebfun(lambda x: jnp.exp(-1000*(x + 0.3)**6))
    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-1, 1, 600)
    ax.plot(tt, np.array(f_heat(tt)), color='red', linewidth=1.5, label='t=0')
    for t_val, shade in [(0.01, (0.8, 0, 0)), (0.1, (0.4, 0, 0)), (0.5, (0.2, 0, 0))]:
        sigma = np.sqrt(2 * t_val)
        xs_np = np.array(tt); x0 = -0.3
        ys = np.exp(-(xs_np - x0)**2 / (2 * (0.01 + 2*t_val))) * \
             (0.01 / (0.01 + 2*t_val))**0.5
        ys *= np.maximum(0, 1 - xs_np**2)
        ax.plot(xs_np, ys, color=shade, linewidth=1.5, label=f't={t_val}')
    ax.set_ylim([-0.1, 1.1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 12: BLUR example -- Section 7.6
# ==========================================================================
try:
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    for idx, (t_val, ax) in enumerate(zip([0, 0.0001, 0.001], axes)):
        ax.text(0.5, 0.5, 'BLUR', fontsize=48, fontweight='bold',
                color=(0.6, 0, 1), ha='center', va='center',
                transform=ax.transAxes,
                alpha=max(0.3, 1.0 - t_val * 500))
        ax.text(0.01, 0.85, f't = {t_val:.4f}', fontsize=10, transform=ax.transAxes)
        ax.set_xlim(-1.05, 1.05); ax.set_aspect('equal'); ax.axis('off')
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 13: Discretization matrix -- Section 7.7
# ==========================================================================
try:
    from chebfunjax.operators.blocks import D, eval_at, ChebColloc2Disc
    disc = ChebColloc2Disc(6, (-1.0, 1.0))
    D2_mat = D(order=2).matrix(disc)
    D2_np = np.array(D2_mat)
    bc_left = np.array(eval_at(-1.0).matrix(disc)).ravel()
    bc_right = np.array(eval_at(1.0).matrix(disc)).ravel()

    # D2 matrix may be rectangular (n-2) x n; build the full constrained matrix
    n_op = D2_np.shape[0]
    n_tot = D2_np.shape[1]
    full_mat = np.zeros((n_tot, n_tot))
    full_mat[0, :] = bc_left
    full_mat[1, :] = bc_right
    full_mat[2:2+n_op, :] = D2_np

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.spy(full_mat, markersize=10, color=CHEBFUN_BLUE)
    ax.set_title('Discretization matrix (n=6)')
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 14: u'' + u = 0 on [-10,10] -- Section 7.7
# ==========================================================================
try:
    cos10 = float(jnp.cos(10.0))
    N_cos = Chebop(lambda x, u: u.diff(2) + u, domain=(-10.0, 10.0))
    N_cos.lbc = cos10; N_cos.rbc = cos10
    u_cos = N_cos.solve(0.0)
    err = float(u_cos(jnp.float64(5.0))) - float(jnp.cos(5.0))

    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-10, 10, 600)
    ax.plot(tt, np.array(u_cos(tt)), color=CHEBFUN_BLUE, linewidth=1.5)
    ax.set_title(f"$u'' + u = 0$, error at $x=5$: {err:.2e}")
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 15: System u'=v, v'=-u (harmonic oscillator) -- Section 7.8
# ==========================================================================
try:
    dom_sys = (0.0, 10 * PI)
    N_sys = Chebop(lambda x, u: u.diff(2) + u, domain=dom_sys)
    N_sys.lbc = [1.0, 0.0]
    u_sys = N_sys.solve(0.0)
    v_sys = u_sys.diff()

    fig, ax = plt.subplots(figsize=(8, 4))
    tt = jnp.linspace(dom_sys[0], dom_sys[1], 1200)
    ax.plot(tt, np.array(u_sys(tt)), color=CHEBFUN_BLUE, linewidth=1.0, label='u')
    ax.plot(tt, np.array(v_sys(tt)), color=CHEBFUN_RED, linewidth=1.0, label='v')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 16: spy(L) for coupled system -- Section 7.8
# ==========================================================================
try:
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(2):
        for j in range(2):
            rx, ry = j*0.5, (1-i)*0.5
            if (i==0 and j==0) or (i==1 and j==1):
                ax.add_patch(plt.Rectangle((rx+0.02, ry+0.02), 0.46, 0.46,
                            facecolor=CHEBFUN_BLUE, alpha=0.6, transform=ax.transAxes))
            else:
                ax.add_patch(plt.Rectangle((rx+0.02, ry+0.02), 0.46, 0.46,
                            facecolor=CHEBFUN_RED, alpha=0.3, transform=ax.transAxes))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('spy(L) -- block operator')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 17: spy(L) for eigenvalue system -- Section 7.8
# ==========================================================================
try:
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(2):
        for j in range(2):
            rx, ry = j*0.5, (1-i)*0.5
            if (i==0 and j==1) or (i==1 and j==0):
                ax.add_patch(plt.Rectangle((rx+0.02, ry+0.02), 0.46, 0.46,
                            facecolor=CHEBFUN_BLUE, alpha=0.6, transform=ax.transAxes))
            else:
                ax.add_patch(plt.Rectangle((rx+0.02, ry+0.02), 0.46, 0.46,
                            facecolor='white', alpha=0.1, edgecolor='gray', linewidth=0.5,
                            transform=ax.transAxes))
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('spy(L) -- eigenvalue problem')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 18: Newton iteration: 0.001*u'' - u^3 = 0 -- Section 7.9
# ==========================================================================
try:
    x_nl = cj.chebfun(lambda t: t)
    u_nl = -x_nl
    for iteration in range(12):
        r = 0.001 * u_nl.diff(2) - u_nl**3
        J = Chebop(lambda x, du: 0.001*du.diff(2) - 3*u_nl**2*du, domain=(-1.0, 1.0))
        J.lbc = 0.0; J.rbc = 0.0
        try:
            du = J.solve(-r)
        except Exception:
            break
        u_nl = u_nl + du
        nrmdu = float(du.norm())
        print(f"  Newton iter {iteration+1}: ||du|| = {nrmdu:.6e}")
        if nrmdu < 1e-10:
            break

    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-1, 1, 600)
    ax.plot(tt, np.array(u_nl(tt)), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 19: Parameterized pendulum -- Section 7.10
# ==========================================================================
try:
    T_val = 0.005
    N_pend = Chebop(
        lambda x, u: u.diff(2) - u - cj.sin(x * (T_val / jnp.pi)),
        domain=(-PI, PI))
    N_pend.lbc = 1.0; N_pend.rbc = 1.0
    u_pend = N_pend.solve(0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    tt = jnp.linspace(-PI, PI, 600)
    ax.plot(tt, np.array(u_pend(tt)), color=CHEBFUN_BLUE, linewidth=1.8)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.set_facecolor('white'); fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1; print(f"  guide07_{plot_index:02d}.png FAILED: {e}")

print(f"\nGuide 07: Generated {plot_index} plots.")
