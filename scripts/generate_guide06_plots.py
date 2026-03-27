"""Generate all plots for Guide Chapter 6: Quasimatrices and Least-Squares."""
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
from chebfunjax.chebfun1d.linalg import Quasimatrix, qr_quasimatrix, svd_quasimatrix
from chebfunjax.domain import Domain

chebfun_style()

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images', 'guide')
os.makedirs(OUTDIR, exist_ok=True)

plot_index = 0

def save(fig):
    global plot_index
    plot_index += 1
    path = os.path.join(OUTDIR, f"guide06_{plot_index:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  guide06_{plot_index:02d}.png saved")

# Build common objects
x = cj.chebfun(lambda t: t)
cols = [x**k for k in range(6)]
A = Quasimatrix(cols, domain=Domain((-1.0, 1.0)))
_colors6 = [CHEBFUN_BLUE, CHEBFUN_RED, '#228B22', '#E08030', '#8B008B', '#008080']
tt = jnp.linspace(-1, 1, 600)

# ==========================================================================
# Plot 1: Columns of A = [1, x, x^2, ..., x^5]  -- Section 6.1
# ==========================================================================
try:
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(6):
        ax.plot(tt, np.array(A[k](tt)), color=_colors6[k], linewidth=1.5)
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 2: spy(A) and spy(A') -- Section 6.1
# ==========================================================================
try:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # spy(A) -- inf x 6
    ax = axes[0]
    for j in range(6):
        ax.plot([j+0.5, j+0.5], [0.1, 0.9], color=CHEBFUN_BLUE, linewidth=4)
    ax.set_xlim([0, 7])
    ax.set_ylim([0, 1])
    ax.set_title('A')
    ax.set_xticks(np.arange(0.5, 6.5, 1))
    ax.set_xticklabels(range(1, 7))
    ax.set_yticks([])
    ax.set_ylabel(r'$\infty$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # spy(A') -- 6 x inf
    ax = axes[1]
    for j in range(6):
        ax.plot([0.1, 0.9], [j+0.5, j+0.5], color=CHEBFUN_BLUE, linewidth=4)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 7])
    ax.set_title("A'")
    ax.set_yticks(np.arange(0.5, 6.5, 1))
    ax.set_yticklabels(range(1, 7))
    ax.set_xticks([])
    ax.set_xlabel(r'$\infty$')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 3: f and its degree-5 least-squares fit  -- Section 6.2
# ==========================================================================
try:
    f = cj.chebfun(lambda t: jnp.exp(t) * jnp.sin(6 * t))
    Q, R = qr_quasimatrix(A)
    rhs = jnp.array([float(Q[j].inner(f)) for j in range(6)])
    c = jnp.linalg.solve(R, rhs)
    ffit = cols[0] * float(c[0])
    for j in range(1, 6):
        ffit = ffit + cols[j] * float(c[j])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, np.array(f(tt)), color=CHEBFUN_BLUE, linewidth=1.5, label='f')
    ax.plot(tt, np.array(ffit(tt)), color=CHEBFUN_RED, linewidth=1.5, label='ffit')
    ax.legend()
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 4: Hat functions  -- Section 6.2
# ==========================================================================
try:
    hat_cols = []
    for j in range(11):
        xj = -1.0 + j / 5.0
        hat_cols.append(
            cj.chebfun(
                lambda t, _xj=xj: jnp.maximum(0.0, 1.0 - 5.0 * jnp.abs(t - _xj)),
                domain=(-1.0, 1.0),
            )
        )
    A2 = Quasimatrix(hat_cols, domain=Domain((-1.0, 1.0)))

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(11):
        ax.plot(tt, np.array(A2[k](tt)), linewidth=1.2)
    ax.set_xticks(np.arange(-1, 1.2, 0.2))
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 5: Hat-function least-squares fit  -- Section 6.2
# ==========================================================================
try:
    Q2_ls, R2_ls = qr_quasimatrix(A2)
    rhs2 = jnp.array([float(Q2_ls[j].inner(f)) for j in range(11)])
    c2 = jnp.linalg.solve(R2_ls, rhs2)
    ffit2 = hat_cols[0] * float(c2[0])
    for j in range(1, 11):
        ffit2 = ffit2 + hat_cols[j] * float(c2[j])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, np.array(f(tt)), color=CHEBFUN_BLUE, linewidth=1.5, label='f')
    ax.plot(tt, np.array(ffit2(tt)), color=CHEBFUN_RED, linewidth=1.5, label='ffit')
    ax.legend()
    ax.set_xticks(np.arange(-1, 1.2, 0.2))
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 6: QR orthonormal columns (L2-normalized Legendre polynomials) -- Sec 6.3
# ==========================================================================
try:
    Q_mono, R_mono = qr_quasimatrix(A)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(Q_mono.n_cols):
        ax.plot(tt, np.array(Q_mono[k](tt)), color=_colors6[k], linewidth=1.5)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 7: spy(A), spy(Q), spy(R)  -- Section 6.3
# ==========================================================================
try:
    R_np = np.array(R_mono)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # spy(A)
    ax = axes[0]
    ax.set_title('A')
    for j in range(6):
        ax.plot([j, j], [0, 1], color=CHEBFUN_BLUE, linewidth=4)
    ax.set_xlim([-0.5, 5.5]); ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(range(6)); ax.set_yticks([])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # spy(Q)
    ax = axes[1]
    ax.set_title('Q')
    for j in range(6):
        ax.plot([j, j], [0, 1], color=CHEBFUN_BLUE, linewidth=4)
    ax.set_xlim([-0.5, 5.5]); ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(range(6)); ax.set_yticks([])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # spy(R) -- upper triangular
    ax = axes[2]
    ax.set_title('R')
    for i in range(6):
        for j in range(6):
            if abs(R_np[i, j]) > 1e-14:
                ax.plot(j, i, 's', color=CHEBFUN_BLUE, markersize=8)
    ax.set_xlim([-0.5, 5.5]); ax.set_ylim([-0.5, 5.5])
    ax.invert_yaxis(); ax.set_aspect('equal')
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 8: Renormalized Legendre (P(1)=1) -- Section 6.3
# ==========================================================================
try:
    Q_leg = []
    for j in range(6):
        val_at_1 = float(Q_mono[j](jnp.float64(1.0)))
        if abs(val_at_1) > 1e-14:
            Q_leg.append(Q_mono[j] * (1.0 / val_at_1))
        else:
            Q_leg.append(Q_mono[j])

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(6):
        ax.plot(tt, np.array(Q_leg[k](tt)), color=_colors6[k], linewidth=1.5)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 9: Orthonormalized hat functions -- Section 6.3
# ==========================================================================
try:
    Q2_hat, R2_hat = qr_quasimatrix(A2)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(Q2_hat.n_cols):
        ax.plot(tt, np.array(Q2_hat[k](tt)), linewidth=1.2)
    ax.set_xticks(np.arange(-1, 1.2, 0.2))
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 10: spy(A), spy(U), spy(S), spy(V) -- Section 6.4
# ==========================================================================
try:
    U_svd, S_svd, V_svd = svd_quasimatrix(A)
    V_np = np.array(V_svd)
    S_np = np.array(S_svd)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for idx_a, (title_, is_inf) in enumerate([('A', True), ('U', True), ('S', False), ('V', False)]):
        ax = axes[idx_a]
        ax.set_title(title_)
        if is_inf:
            for j in range(6):
                ax.plot([j, j], [0, 1], color=CHEBFUN_BLUE, linewidth=4)
            ax.set_xlim([-0.5, 5.5]); ax.set_ylim([-0.1, 1.1])
            ax.set_xticks(range(6)); ax.set_yticks([])
        elif title_ == 'S':
            for i in range(6):
                ax.plot(i, i, 's', color=CHEBFUN_BLUE, markersize=8)
            ax.set_xlim([-0.5, 5.5]); ax.set_ylim([-0.5, 5.5])
            ax.invert_yaxis(); ax.set_aspect('equal')
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
        else:  # V
            for i in range(6):
                for j in range(6):
                    if abs(V_np[i, j]) > 1e-14:
                        ax.plot(j, i, 's', color=CHEBFUN_BLUE, markersize=8)
            ax.set_xlim([-0.5, 5.5]); ax.set_ylim([-0.5, 5.5])
            ax.invert_yaxis(); ax.set_aspect('equal')
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 11: Extremal functions A*v1 (blue) and A*vn (red) -- Section 6.4
# ==========================================================================
try:
    v1 = np.array(V_svd[:, 0])
    vn = np.array(V_svd[:, -1])
    f_max = sum(float(v1[j]) * cols[j] for j in range(6))
    f_min = sum(float(vn[j]) * cols[j] for j in range(6))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tt, np.array(f_max(tt)), color=CHEBFUN_BLUE, linewidth=1.5)
    ax.plot(tt, np.array(f_min(tt)), color=CHEBFUN_RED, linewidth=1.5)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

# ==========================================================================
# Plot 12: spy(null(B)), spy(orth(B)), spy(pinv(A)) -- Section 6.6
# ==========================================================================
try:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # null(B) -- 3 x 1 vector
    ax = axes[0]
    ax.set_title('null(B)')
    for i in range(3):
        ax.plot(0, i, 's', color=CHEBFUN_BLUE, markersize=10)
    ax.set_xlim([-0.5, 0.5]); ax.set_ylim([-0.5, 2.5])
    ax.invert_yaxis()
    ax.set_xticks([0]); ax.set_yticks(range(3))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # orth(B) -- inf x 2
    ax = axes[1]
    ax.set_title('orth(B)')
    for j in range(2):
        ax.plot([j, j], [0, 1], color=CHEBFUN_BLUE, linewidth=4)
    ax.set_xlim([-0.5, 1.5]); ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(range(2)); ax.set_yticks([])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # pinv(A) -- 6 x inf
    ax = axes[2]
    ax.set_title('pinv(A)')
    for j in range(6):
        ax.plot([0, 1], [j, j], color=CHEBFUN_BLUE, linewidth=3)
    ax.set_xlim([-0.1, 1.1]); ax.set_ylim([-0.5, 5.5])
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks(range(6))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.set_facecolor('white')
    fig.tight_layout()
    save(fig)
except Exception as e:
    plot_index += 1
    print(f"  guide06_{plot_index:02d}.png FAILED: {e}")

print(f"\nGuide 06: Generated {plot_index} plots.")
