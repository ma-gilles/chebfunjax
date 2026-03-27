"""Lyapunov exponents of the Lorenz system.

Computes the maximal Lyapunov exponent of the Lorenz system by tracking
the growth of a perturbation and periodically renormalizing.

Credit: Chebfun example ode-nonlin/LyapunovExponents.m (Hrothgar, Jan 2015).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()



def run():
    print("=" * 60)
    print("Maximal Lyapunov exponent of the Lorenz system")
    print("=" * 60)

    sigma, rho, beta = 10.0, 28.0, 8.0/3.0

    def lorenz(t, state):
        x, y, z = state
        return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

    def lorenz_tangent(t, state):
        """Lorenz + tangent vector [dx, dy, dz, d(dx), d(dy), d(dz)]."""
        x, y, z, dx, dy, dz = state
        # Lorenz
        fx = sigma*(y-x)
        fy = x*(rho-z)-y
        fz = x*y-beta*z
        # Jacobian J applied to tangent [dx,dy,dz]
        # J = [[-sigma, sigma, 0], [rho-z, -1, -x], [y, x, -beta]]
        dfx = -sigma*dx + sigma*dy
        dfy = (rho-z)*dx - dy - x*dz
        dfz = y*dx + x*dy - beta*dz
        return [fx, fy, fz, dfx, dfy, dfz]

    T_total = 50.0
    T_renorm = 0.5
    n_steps = int(T_total / T_renorm)

    ic = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    ic[3:] /= np.linalg.norm(ic[3:])  # normalize initial tangent

    state = ic.copy()
    log_growth = 0.0
    lyapunov_running = []
    t_cumulative = 0.0

    print(f"\nComputing with renormalization every T={T_renorm}s...")
    for k in range(n_steps):
        sol = solve_ivp(lorenz_tangent, [0, T_renorm], state,
                        rtol=1e-10, atol=1e-12)
        state = sol.y[:, -1].copy()
        norm_tangent = np.linalg.norm(state[3:])
        log_growth += np.log(norm_tangent)
        state[3:] /= norm_tangent  # renormalize
        t_cumulative += T_renorm
        lyap_est = log_growth / t_cumulative
        lyapunov_running.append((t_cumulative, lyap_est))

    lambda_max = lyapunov_running[-1][1]
    print(f"\nMaximal Lyapunov exponent: λ₁ ≈ {lambda_max:.4f}")
    print(f"  Literature value: λ₁ ≈ 0.906 (σ=10, ρ=28, β=8/3)")

    # Check that it's positive (chaotic system)
    assert lambda_max > 0.5, f"Lyapunov exponent too small: {lambda_max}"
    assert lambda_max < 1.5, f"Lyapunov exponent too large: {lambda_max}"

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    t_run = [x[0] for x in lyapunov_running]
    lam_run = [x[1] for x in lyapunov_running]

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(t_run, lam_run, 'b', linewidth=1.4)
    axes[0].axhline(lambda_max, color='r', linestyle='--', linewidth=1.0,
                    label=f"λ₁ ≈ {lambda_max:.4f}")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("λ₁(t)")
    axes[0].set_title("Running Lyapunov exponent estimate", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Also show attractor x-z projection
    sol_viz = solve_ivp(lorenz, [0, 30], [1,1,1],
                        t_eval=np.linspace(0, 30, 5000), rtol=1e-10)
    axes[1].plot(sol_viz.y[0, ::2], sol_viz.y[2, ::2], 'b', linewidth=0.5, alpha=0.5)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("z")
    axes[1].set_title("Lorenz attractor (x-z plane)", fontsize=10)
    axes[1].grid(True, alpha=0.2)

    fig.suptitle(f"Lorenz Lyapunov exponent λ₁ ≈ {lambda_max:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "lyapunov_exponents.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
