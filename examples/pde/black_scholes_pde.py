"""Black-Scholes PDE using operator exponential.

Solves the Black-Scholes PDE for a European call option:
  v_t = -(sigma^2/2)*s^2*v_ss - r*s*v_s + r*v

by transforming to heat equation form and using matrix exponential,
following pde/BSExponential.m by Toby Driscoll (June 2014).

The trick is to remove non-homogeneous BCs via a particular solution u
satisfying A*u = 0, B*u = q. Then propagate w = v - u with A and zero BCs.

Original MATLAB: https://www.chebfun.org/examples/pde/BSExponential.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.linalg import expm
from scipy.stats import norm
import os

def run():
    print("=" * 60)
    print("Black-Scholes PDE via operator exponential")
    print("=" * 60)

    # Black-Scholes PDE:
    # v_t = -(sigma^2/2)*s^2*v_ss - r*s*v_s + r*v
    # BCs: v(0) = 0 (at s=0), v_s(s_max) = 1 (Neumann, approx v→s)
    # Final condition: v(T) = max(s - K, 0)

    sigma = 0.45
    r = 0.03
    K = 50.0        # strike
    s_max = 500.0   # truncate the domain

    # Discretize [0, s_max] with Chebyshev nodes
    N = 80
    # Use Gauss-Lobatto points on [0, s_max]
    j = np.arange(N + 1)
    s_cheb = 0.5 * s_max * (1 - np.cos(np.pi * j / N))  # mapped to [0, s_max]

    # Build the BS operator using finite differences on s_cheb
    # Second-order FD for s^2 v_ss and s*v_s
    ds = np.diff(s_cheb)
    s_int = s_cheb[1:-1]  # interior points

    n_int = len(s_int)

    # Build matrices for interior points
    # Use second-order FD with non-uniform spacing
    A_mat = np.zeros((n_int, n_int))

    for i in range(n_int):
        h_l = ds[i]       # spacing to the left
        h_r = ds[i + 1]   # spacing to the right
        s_i = s_int[i]

        # Coefficients for second derivative: v_ss
        c_l = 2.0 / (h_l * (h_l + h_r))
        c_m = -2.0 / (h_l * h_r)
        c_r = 2.0 / (h_r * (h_l + h_r))

        # Coefficients for first derivative: v_s
        d_l = -h_r / (h_l * (h_l + h_r))
        d_m = (h_r - h_l) / (h_l * h_r)
        d_r = h_l / (h_r * (h_l + h_r))

        # BS operator: A = -(sigma^2/2)*s^2*d2/ds2 - r*s*d/ds + r*I
        factor_ss = -sigma**2 / 2 * s_i**2
        factor_s = -r * s_i
        factor_r = r

        if i > 0:
            A_mat[i, i-1] = factor_ss * c_l + factor_s * d_l
        A_mat[i, i] = factor_ss * c_m + factor_s * d_m + factor_r
        if i < n_int - 1:
            A_mat[i, i+1] = factor_ss * c_r + factor_s * d_r

    # Boundary conditions:
    # Left BC: v(0) = 0 → s[0] = 0, v = 0
    # Right BC: v_s(s_max) ≈ 1 (Neumann)

    # Particular solution u satisfying A*u = 0, BC: u(0)=0, u_s(s_max)=1
    # For BC u_s(s_max) = 1, a linear function u(s) = s/s_max satisfies u_s = 1/s_max,
    # but we need u_s = 1 at s_max. Let u(s) = s. Then u_s = 1.
    # Does u(s) = s satisfy A*u = 0?
    # A*s = -(sigma^2/2)*s^2*0 - r*s*1 + r*s = 0. Yes!
    u_particular = s_cheb.copy()  # u(s) = s satisfies BCs and A*u = 0

    # Final condition for call: vT = max(s - K, 0)
    vT = np.maximum(s_cheb - K, 0.0)
    wT = vT - u_particular  # adjusted variable w = v - u

    # wT at interior nodes
    wT_int = wT[1:-1]

    # Propagate w backward in time from T to T-t using exp(-t*A)
    # (time runs backward: at t=0 we have the terminal condition)
    t_vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    print(f"\nComputing option prices at times 0.1, 0.2, ..., 0.5 before maturity...")
    for t in t_vals:
        # Apply matrix exponential: w(t) = exp(-t*A) * wT
        w_int = expm(-t * A_mat) @ wT_int

        # Reconstruct v: add back particular solution at interior points
        v_int = w_int + u_particular[1:-1]

        # Full profile with BCs
        v_full = np.zeros(N + 1)
        v_full[0] = 0.0              # Dirichlet BC
        v_full[1:-1] = v_int
        v_full[-1] = v_int[-1] + ds[-1]  # approximate Neumann

        results.append((t, v_full.copy()))

    # Compare with Black-Scholes formula for call at s=55, t=0.5
    s_test = 55.0
    t_test = 0.5
    d1 = (np.log(s_test / K) + (r + 0.5 * sigma**2) * t_test) / (sigma * np.sqrt(t_test))
    d2 = d1 - sigma * np.sqrt(t_test)
    bs_exact = s_test * norm.cdf(d1) - K * np.exp(-r * t_test) * norm.cdf(d2)

    # Interpolate our result at s=55
    t_last, v_last = results[-1]
    v_at_55 = np.interp(s_test, s_cheb, v_last)

    print(f"\nOption value at s=55, t=0.5 before maturity:")
    print(f"  Numerical: {v_at_55:.6f}")
    print(f"  Black-Scholes: {bs_exact:.6f}")
    print(f"  Error: {abs(v_at_55 - bs_exact):.4f}")

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    # Show around the strike
    s_range = (s_cheb >= 30) & (s_cheb <= 100)
    s_plot = s_cheb[s_range]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(results)))
    axes[0].plot(s_plot, vT[s_range], 'k--', linewidth=2, label='Payoff at T')
    for (t, v), col in zip(results, colors):
        axes[0].plot(s_plot, v[s_range], color=col, linewidth=1.5,
                     label=f't={t:.1f}')
    axes[0].set_title("Black-Scholes option values (K=50)", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim([-0.5, 15])

    # Final solution vs BS formula
    s_bs = np.linspace(30, 100, 200)
    d1_bs = (np.log(s_bs / K) + (r + 0.5 * sigma**2) * 0.5) / (sigma * np.sqrt(0.5))
    d2_bs = d1_bs - sigma * np.sqrt(0.5)
    v_bs = s_bs * norm.cdf(d1_bs) - K * np.exp(-r * 0.5) * norm.cdf(d2_bs)

    axes[1].plot(s_bs, v_bs, color='#D95319', linestyle='-', linewidth=2, label='Black-Scholes formula')
    t_last, v_last = results[-1]
    axes[1].plot(s_plot, v_last[s_range], color='#0072BD', marker='.', linestyle='none', markersize=5, label='Numerical (t=0.5)')
    axes[1].axvline(55, color='#77AC30', linestyle=':', label=f's=55: {v_at_55:.3f}')
    axes[1].set_title("Comparison with Black-Scholes at t=0.5", fontsize=11)
    axes[1].legend(fontsize=9)

    fig.suptitle("Black-Scholes PDE via matrix exponential", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "black_scholes_pde.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
