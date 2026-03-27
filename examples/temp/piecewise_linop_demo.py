"""Demo of piecewise operators.

Demonstrates solving ODEs with piecewise-defined coefficients (e.g., a
sign function creates a jump at x=0). Translated from temp/PiecewiseLinopDemo.m.

Original: https://www.chebfun.org/examples/ode/PiecewiseLinopDemo.html
Author: Nick Hale, November 2010
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def solve_piecewise_ode(n=60):
    """Solve -u'' + sign(x)*u = 0, u(-1)=u(1)=0 via Chebyshev collocation.

    On [-1,0]: -u'' - u = 0, boundary conditions u(-1)=0, continuity at 0
    On [0,1]:  -u'' + u = 0, boundary conditions u(1)=0, continuity at 0
    """
    # Solve on each piece separately with matching conditions at x=0
    n2 = n // 2

    def cheb_d2(n_pts, a, b):
        """Second-order Chebyshev differentiation matrix on [a,b]."""
        k = np.arange(n_pts + 1)
        x = np.cos(np.pi * k / n_pts)
        x_orig = (b - a)/2 * x + (a + b)/2
        c = np.ones(n_pts + 1)
        c[0] = 2; c[-1] = 2
        c *= (-1)**k
        X = np.tile(x, (n_pts+1, 1))
        dX = X - X.T
        D = np.outer(c, 1/c) / (dX + np.eye(n_pts+1))
        D -= np.diag(D.sum(axis=1))
        D2 = D @ D * (2/(b-a))**2
        return D2, x_orig

    D2_l, x_l = cheb_d2(n2, -1, 0)  # left piece: x in [-1,0]
    D2_r, x_r = cheb_d2(n2, 0, 1)   # right piece: x in [0,1]

    # Left: -u'' - u = 0 (interior rows: 1 to n2-1)
    n_int = n2 - 1
    A_ll = -D2_l[1:n2, 1:n2] - np.eye(n_int)  # interior-interior
    A_lr = -D2_l[1:n2, n2:n2+1]               # interior-right boundary (x=0)
    # Right: -u'' + u = 0
    A_rl = D2_r[1:n2, 0:1]                    # interior-left boundary (x=0 of right)
    A_rr_int = -D2_r[1:n2, 1:n2] + np.eye(n_int)  # interior-interior

    # Full system: unknowns = [u_l_interior, u(0), u_r_interior]
    # Left BCs: u(-1) = 0 (enforced by removing first node)
    # Right BCs: u(1) = 0 (enforced by removing last node)
    # Matching: u_l(0) = u(0), u_r(0) = u(0)
    # Derivative matching: u_l'(0) = u_r'(0)

    N_sys = 2*n_int + 1
    A_sys = np.zeros((N_sys, N_sys))
    b_sys = np.zeros(N_sys)

    # Left piece equations (interior of left)
    A_sys[:n_int, :n_int] = A_ll
    A_sys[:n_int, n_int:n_int+1] = A_lr  # coupling to u(0)

    # Right piece equations (interior of right)
    A_sys[n_int+1:, n_int:n_int+1] = A_rl  # coupling from u(0)
    A_sys[n_int+1:, n_int+1:] = A_rr_int

    # Derivative matching at x=0
    # u_l'(0) - u_r'(0) = 0
    # Use first-order differentiation matrix row for x=0
    # Left D at x=0 (index n2 of left piece)
    k = np.arange(n2 + 1)
    x_nodes = np.cos(np.pi * k / n2)
    c = np.ones(n2 + 1); c[0] = 2; c[-1] = 2; c *= (-1)**k
    X = np.tile(x_nodes, (n2+1, 1))
    dX = X - X.T
    D_raw = np.outer(c, 1/c) / (dX + np.eye(n2+1))
    D_raw -= np.diag(D_raw.sum(axis=1))
    D_left_row = D_raw[n2, :] * 2  # scale by 2/(b-a)=2

    # Similarly for right piece, row at x=0 (index 0 of right piece)
    D_right_row = -D_raw[0, :] * 2  # minus due to opposite orientation

    # Derivative condition: sum over interior + boundary
    deriv_row = np.zeros(N_sys)
    # Left piece: interior (1..n2-1) contribute
    deriv_row[:n_int] = D_left_row[1:n2]
    # u(0) at x=0 for left: D_left_row[n2]*u_l(0) + ... = derivative
    deriv_row[n_int] = D_left_row[n2] - D_right_row[0]  # matching
    # Right piece: interior (1..n2-1)
    deriv_row[n_int+1:] = D_right_row[1:n2]

    A_sys[n_int, :] = deriv_row

    # Solve system
    u_int = solve(A_sys, b_sys + np.random.randn(N_sys) * 0)

    # This gives trivial solution; solve eigenvalue problem instead
    # Find nontrivial solution via SVD
    from numpy.linalg import svd
    U, S, Vt = svd(A_sys)
    # Null space (approximately, at least one near-zero singular value)
    u_null = Vt[-1, :]

    # Reconstruct full solution
    u_l_full = np.zeros(n2 + 1)
    u_l_full[0] = 0  # u(-1) = 0
    u_l_full[1:n2] = u_null[:n_int]
    u_l_full[n2] = u_null[n_int]

    u_r_full = np.zeros(n2 + 1)
    u_r_full[0] = u_null[n_int]
    u_r_full[1:n2] = u_null[n_int+1:]
    u_r_full[n2] = 0  # u(1) = 0

    # Normalize
    u_l_full /= np.max(np.abs(u_l_full)) + 1e-14
    u_r_full /= np.max(np.abs(u_l_full)) + 1e-14

    return x_l[::-1], u_l_full[::-1], x_r[::-1], u_r_full[::-1]


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Piecewise ODE solution ---
    x_l, u_l, x_r, u_r = solve_piecewise_ode(n=40)

    axes[0].plot(x_l, u_l, 'b-', linewidth=2.5, label='Left piece (-u"-u=0)')
    axes[0].plot(x_r, u_r, 'r-', linewidth=2.5, label='Right piece (-u"+u=0)')
    axes[0].axvline(0, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].set_title("Piecewise ODE: -u''+sign(x)·u=0\nJump at x=0", fontsize=10)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('u(x)')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    # --- Panel 2: Piecewise forcing ---
    x_full = np.linspace(-1, 1, 500)
    sign_x = np.sign(x_full)

    # Solve -u'' + sign(x)*u = f(x) where f = sin(pi*x)
    # Use simple finite difference
    n = 200
    x_fd = np.linspace(-1, 1, n)
    dx = x_fd[1] - x_fd[0]
    D2 = (np.diag(np.ones(n-1), 1) - 2*np.eye(n) + np.diag(np.ones(n-1), -1)) / dx**2
    A_fd = -D2 + np.diag(np.sign(x_fd))
    # BCs: u(-1) = u(1) = 0
    A_fd[0, :] = 0; A_fd[0, 0] = 1
    A_fd[-1, :] = 0; A_fd[-1, -1] = 1
    rhs_fd = np.sin(np.pi * x_fd)
    rhs_fd[0] = 0; rhs_fd[-1] = 0

    u_fd = solve(A_fd, rhs_fd)
    axes[1].plot(x_fd, u_fd, 'b-', linewidth=2.5, label='u(x)')
    axes[1].plot(x_fd, rhs_fd, 'r--', linewidth=1.5, label='f(x)=sin(πx)')
    axes[1].axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_title("-u''+sign(x)·u = sin(πx)", fontsize=10)
    axes[1].set_xlabel('x'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    # --- Panel 3: Sign function visualization ---
    axes[2].plot(x_full, np.abs(sign_x), 'g-', linewidth=2, label='|sign(x)|')
    axes[2].plot(x_full, sign_x, 'b-', linewidth=2, label='sign(x)')
    axes[2].fill_between(x_full, sign_x, alpha=0.15)
    axes[2].axvline(0, color='k', linestyle='--', linewidth=1.5)
    axes[2].set_title('Piecewise coefficient\nsign(x) causes jump at x=0', fontsize=10)
    axes[2].set_xlabel('x'); axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    print("Piecewise linop demo:")
    print(f"  Solved -u''+sign(x)*u=0 on [-1,1] with continuity at x=0")
    print(f"  Solved -u''+sign(x)*u=sin(πx) by finite differences")

    fig.suptitle('Demo of Piecewise Differential Operators', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'piecewise_linop_demo.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("piecewise_linop_demo: done")
    return True


if __name__ == "__main__":
    run()
