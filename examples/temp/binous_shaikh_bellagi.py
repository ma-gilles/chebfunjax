"""Problems from Binous, Shaikh and Bellagi.

Solves several nonlinear ODEs and PDEs from Binous, Shaikh and Bellagi
using numpy/scipy, demonstrating Chebyshev collocation methods.
Translated from temp/BinousShaikhBellagi.m.

Original: https://www.chebfun.org/examples/temp/BinousShaikhBellagi.html
Author: Nick Trefethen, September 2014
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
import chebfunjax as cj


def cheb_matrix(n):
    """Chebyshev differentiation matrix on [-1,1] with n+1 points."""
    if n == 0:
        return np.array([[0.0]]), np.array([1.0])
    k = np.arange(n + 1)
    x = np.cos(np.pi * k / n)
    c = np.ones(n + 1)
    c[0] = 2; c[-1] = 2
    c *= (-1)**k
    X = np.tile(x, (n+1, 1))
    dX = X - X.T
    D = np.outer(c, 1/c) / (dX + np.eye(n+1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def solve_nonlinear_bvp(n=40):
    """Solve u*u' - u'' = 1, u(-1)=0, u(1)=2 via Newton iteration."""
    D, x = cheb_matrix(n)
    D2 = D @ D

    # Initial guess: linear interpolation
    u = np.linspace(0, 2, n+1)

    for _ in range(30):
        # Residual: u*u' - u'' - 1 (interior only)
        up = D @ u
        res = u * up - D2 @ u - 1
        res_int = res[1:-1]  # interior residual

        # Jacobian: d(res)/du at interior
        J = np.diag(up[1:-1]) + np.diag(u[1:-1]) @ D[1:-1, 1:-1] - D2[1:-1, 1:-1]

        # Newton step
        delta = np.zeros(n+1)
        delta[1:-1] = solve(J, -res_int)

        u = u + delta
        # Enforce BCs
        u[0] = 0; u[-1] = 2

        if np.max(np.abs(delta[1:-1])) < 1e-12:
            break

    return x, u


def heat_equation_solution(t_final=0.0126):
    """Solve u_t = u_yy, u(0)=u(1)=0, u(0,x)=1 using mode expansion."""
    # u(t,x) = sum_n a_n * sin(n*pi*x) * exp(-n^2*pi^2*t)
    y = np.linspace(0, 1, 500)
    n_max = 50
    u = np.zeros_like(y)
    for n in range(1, n_max, 2):  # only odd terms for initial condition 1
        an = 4 / (n * np.pi)
        u += an * np.sin(n * np.pi * y) * np.exp(-n**2 * np.pi**2 * t_final)
    return y, u


def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/temp')
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Panel 1: Nonlinear BVP u*u' - u'' = 1 ---
    x, u = solve_nonlinear_bvp(n=40)

    axes[0].plot(x, u, 'b-', linewidth=2.5)
    axes[0].axhline(u[len(u)//2], color='g', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_title("Nonlinear BVP: u·u' - u'' = 1\nu(-1)=0, u(1)=2", fontsize=10)
    axes[0].set_xlabel('x'); axes[0].set_ylabel('u(x)')
    axes[0].grid(True, alpha=0.3)

    # Values at x=0 and x=1/sqrt(2)
    # Interpolate
    from scipy.interpolate import interp1d
    u_interp = interp1d(x, u, kind='cubic')
    val_0 = u_interp(0.0)
    val_sqrt2 = u_interp(1/np.sqrt(2))
    print(f"Problem 1 (u*u' - u'' = 1):")
    print(f"  u(0) = {val_0:.6f}")
    print(f"  u(1/√2) = {val_sqrt2:.6f}")
    axes[0].plot(0, val_0, 'ro', markersize=8, label=f'u(0)={val_0:.3f}')
    axes[0].plot(1/np.sqrt(2), val_sqrt2, 'gs', markersize=8,
                 label=f'u(1/√2)={val_sqrt2:.3f}')
    axes[0].legend(fontsize=9)

    # --- Panel 2: Diffusion equation ---
    y, u_heat = heat_equation_solution(t_final=0.0126)

    axes[1].plot(y, u_heat, 'r-', linewidth=2.5, label='t=0.0126')
    # Initial condition
    axes[1].axhline(1, color='k', linestyle='--', linewidth=1.5,
                    label='t=0 (IC)')
    axes[1].set_title('Heat equation: u_t = u_yy\nu(t,0)=u(t,1)=0', fontsize=10)
    axes[1].set_xlabel('y'); axes[1].set_ylabel('u(t,y)')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    val_half = np.interp(0.5, y, u_heat)
    print(f"\nProblem 2 (heat equation, t=0.0126):")
    print(f"  u(0.5) = {val_half:.6f}")
    axes[1].plot(0.5, val_half, 'b*', markersize=12,
                 label=f'u(0.5)={val_half:.4f}')
    axes[1].legend(fontsize=9)

    # --- Panel 3: Unsteady convection-diffusion ---
    # c_t = 0.49*D*c_xx - 2.5*c_x, c(t,0)=1, c(t,∞)=0, c(0,z)=0
    # Use similarity solution or numerical approach on truncated domain [0,10]
    D_val = 0.49
    Pe = 2.5 / D_val  # Peclet number

    # Steady state: c = exp(-Pe * z) * constant
    z = np.linspace(0, 5, 500)
    c_steady = np.exp(-Pe * z)

    # Transient solution at various times (approximate)
    from scipy.special import erfc
    times = [0.01, 0.05, 0.1, 0.5]
    colors_t = ['b', 'g', 'r', 'm']
    for t, col in zip(times, colors_t):
        # Ogata-Banks solution
        c_t = (0.5 * erfc((z - 2.5*t) / (2*np.sqrt(D_val*t + 1e-15)))
               + 0.5 * np.exp(Pe*z)
               * erfc((z + 2.5*t) / (2*np.sqrt(D_val*t + 1e-15))))
        axes[2].plot(z, np.clip(c_t, 0, 1), '-', color=col,
                     linewidth=1.5, label=f't={t}')

    axes[2].plot(z, c_steady, 'k--', linewidth=2, label='Steady state')
    axes[2].set_title('Convection-diffusion\nc_t = 0.49·c_xx - 2.5·c_x', fontsize=10)
    axes[2].set_xlabel('z'); axes[2].set_ylabel('c(t,z)')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    fig.suptitle('Problems from Binous, Shaikh and Bellagi', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'binous_shaikh_bellagi.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print("binous_shaikh_bellagi: done")
    return True


if __name__ == "__main__":
    run()
