"""An ODE on an unbounded interval.

Solves 0.1*u'' + u' + u = 0 on [0, inf) with u(0)=1, u(inf)=0,
truncating the domain to [0, L] for a large L.
Translated from temp/UnboundedODE.m.

Original: https://www.chebfun.org/examples/ode/UnboundedODE.html
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

def solve_unbounded_ode(L=30, n=200):
    """Solve 0.1*u'' + u' + u = 0 on [0,L], u(0)=1, u(L)~0."""
    x = np.linspace(0, L, n)
    dx = x[1] - x[0]

    D1 = (np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)) / (2*dx)
    D2 = (np.diag(np.ones(n-1), 1) - 2*np.eye(n) + np.diag(np.ones(n-1), -1)) / dx**2

    A = 0.1 * D2 + D1 + np.eye(n)
    # BCs
    A[0, :] = 0; A[0, 0] = 1
    A[-1, :] = 0; A[-1, -1] = 1
    b = np.zeros(n)
    b[0] = 1; b[-1] = 0

    u = solve(A, b)
    return x, u

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    # Exact solution: characteristic equation 0.1*r^2 + r + 1 = 0
    # r = (-1 ± sqrt(1 - 0.4)) / 0.2 = (-1 ± sqrt(0.6)) / 0.2
    r1 = (-1 + np.sqrt(0.6)) / 0.2
    r2 = (-1 - np.sqrt(0.6)) / 0.2
    # General: u = A*exp(r1*x) + B*exp(r2*x)
    # u(0)=1: A+B=1
    # u(inf)=0: A=0 (since r1>0), so B=1
    # Thus: u(x) = exp(r2*x)
    print(f"ODE 0.1u'' + u' + u = 0:")
    print(f"  Roots: r1={r1:.4f}, r2={r2:.4f}")
    print(f"  Exact solution: u = exp({r2:.4f}*x)")

    x_exact = np.linspace(0, 30, 500)
    u_exact = np.exp(r2 * x_exact)

    fig, axes = plt.subplots(1, 3)

    # --- Panel 1: Numerical vs exact ---
    x_num, u_num = solve_unbounded_ode(L=30, n=300)

    axes[0].plot(x_exact, u_exact, 'k-', linewidth=2, label='Exact exp(r₂x)')
    axes[0].plot(x_num, u_num, 'r--', linewidth=2, label='Numerical')
    axes[0].set_title('0.1u\'\'+u\'+u=0 on [0,∞)\nTruncated to [0,30]', fontsize=10)
    axes[0].legend(fontsize=9)

    err = np.max(np.abs(u_exact - np.interp(x_exact, x_num, u_num)))
    print(f"  Max error (numerical vs exact): {err:.4e}")

    # --- Panel 2: Effect of domain truncation ---
    for L_trunc, col in zip([5, 10, 20, 30], ['b', 'g', 'r', 'm']):
        x_t, u_t = solve_unbounded_ode(L=L_trunc, n=200)
        axes[1].plot(x_t, u_t, '-', color=col, linewidth=1.5,
                     label=f'L={L_trunc}')

    axes[1].plot(x_exact, u_exact, 'k-', linewidth=2.5, label='Exact')
    axes[1].set_title('Effect of domain truncation', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 20)

    # --- Panel 3: Piecewise variant with jump in coefficient ---
    # 0.1*u'' + u' + u - 1.5*(x<2)*u = 0
    x_pw, u_pw = solve_unbounded_ode(L=30, n=300)  # base solution

    # Modified: subtract 1.5*u for x<2
    n = 300
    x_pw2 = np.linspace(0, 30, n)
    dx = x_pw2[1] - x_pw2[0]
    D1 = (np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)) / (2*dx)
    D2 = (np.diag(np.ones(n-1), 1) - 2*np.eye(n) + np.diag(np.ones(n-1), -1)) / dx**2

    piecewise_coeff = np.where(x_pw2 < 2, 1.0 - 1.5, 1.0)
    A2 = 0.1 * D2 + D1 + np.diag(piecewise_coeff)
    A2[0, :] = 0; A2[0, 0] = 1
    A2[-1, :] = 0; A2[-1, -1] = 1
    b2 = np.zeros(n); b2[0] = 1

    u_pw2 = solve(A2, b2)

    axes[2].plot(x_exact, u_exact, 'k-', linewidth=2, label='Original')
    axes[2].plot(x_pw2, u_pw2, 'r-', linewidth=2, label='Piecewise (x<2)')
    axes[2].axvline(2, color='g', linestyle='--', linewidth=1.5,
                     label='Breakpoint x=2')
    axes[2].set_title('Piecewise variant:\n1.5 subtracted for x<2', fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].set_xlim(0, 20)

    fig.suptitle('ODE on Unbounded Interval [0, ∞)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'unbounded_ode.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("unbounded_ode: done")
    return True

if __name__ == "__main__":
    run()
