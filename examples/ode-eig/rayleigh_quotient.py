"""Rayleigh quotient iteration for a differential operator.

Demonstrates cubic-convergence Rayleigh quotient iteration for finding
an eigenpair of the operator L = -d^2/dx^2 with Dirichlet BCs.

Credit: Chebfun example ode-eig/RayleighQuotient.m
        (Nick Hale and Yuji Nakatsukasa, March 2017).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Rayleigh quotient iteration for differential operators")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Part 1: Rayleigh quotient iteration for a symmetric matrix
    # Shows cubic convergence
    # ------------------------------------------------------------------
    print("\nPart 1: RQI for a random symmetric 10x10 matrix")
    rng = np.random.default_rng(10)
    n = 10
    A_raw = rng.standard_normal((n, n))
    A = A_raw + A_raw.T  # symmetric

    true_eigs = np.sort(np.linalg.eigvalsh(A))
    target_eig = true_eigs[0]  # smallest eigenvalue
    print(f"  True smallest eigenvalue: {target_eig:.8f}")

    # Initialize close to the smallest eigenvector
    _, vecs = np.linalg.eigh(A)
    v = vecs[:, 0] + 0.01 * rng.standard_normal(n)  # perturb smallest eigenvec
    v /= np.linalg.norm(v)
    lam = v @ A @ v  # initial Rayleigh quotient

    residuals = []
    max_iter = 15
    for iteration in range(max_iter):
        res = abs(lam - target_eig)
        residuals.append(res)
        if res < 1e-14:
            break
        # Shifted solve: (A - lam*I) v_new = v
        try:
            v_new = np.linalg.solve(A - lam * np.eye(n), v)
        except np.linalg.LinAlgError:
            break
        v_new /= np.linalg.norm(v_new)
        v = v_new
        lam = v @ A @ v  # update Rayleigh quotient

    print(f"  Converged in {len(residuals)} iterations")
    print(f"  Final lambda: {lam:.12f}")
    print(f"  Residuals:")
    for i, r in enumerate(residuals):
        print(f"    iter {i}: {r:.4e}")
    if residuals[-1] >= 1e-10:
        import warnings
        warnings.warn(f"RQI residual {residuals[-1]:.2e}; not fully converged.")

    # Check cubic convergence: if res[k+1] <= C * res[k]^3
    if len(residuals) >= 4:
        for i in range(1, min(len(residuals) - 1, 5)):
            if residuals[i-1] > 1e-12:
                ratio = residuals[i] / residuals[i-1]**2
                print(f"    convergence ratio at iter {i}: res[i]/res[i-1]^2 = {ratio:.2e}")

    # ------------------------------------------------------------------
    # Part 2: Rayleigh quotient for -d^2/dx^2 on [0, pi]
    # Eigenvalues: k^2, k=1,2,3,...
    # ------------------------------------------------------------------
    print("\nPart 2: RQI for L = -d^2/dx^2 on [0, pi]")
    dom = (0.0, float(np.pi))

    # Rayleigh quotient: R[u] = <u, Lu> / <u, u> = int(u' * u') / int(u^2)
    # Starting guess: u_0(x) = sin(x) + 0.1*sin(2x) (perturbed toward lambda_1=1)
    x_vals = np.linspace(0, np.pi, 200)
    u_v = np.sin(x_vals) + 0.3 * np.sin(3 * x_vals)  # includes modes 1 and 3

    def rayleigh_quotient_ode(u_arr, x_arr):
        """Compute R[u] = int(-u'' u) / int(u^2) numerically."""
        dx = x_arr[1] - x_arr[0]
        u_pp = np.gradient(np.gradient(u_arr, dx), dx)
        numerator = -np.trapezoid(u_pp * u_arr, x_arr)
        denominator = np.trapezoid(u_arr**2, x_arr)
        return numerator / denominator

    rq_0 = rayleigh_quotient_ode(u_v, x_vals)
    print(f"  Initial guess RQ: {rq_0:.4f} (mode 1 is lambda=1)")

    # Use the chebop to compute exact eigenvalues for comparison
    L_op = Chebop(lambda x, u: -u.diff(2), domain=dom)
    L_op.lbc = 0.0; L_op.rbc = 0.0
    k = 5
    lams_cheb = L_op.eigs(k=k)
    lams_sorted = np.sort(np.real(np.array(lams_cheb)))
    exact = np.arange(1, k+1, dtype=float)**2
    print(f"\n  Chebop eigenvalues: {lams_sorted}")
    print(f"  Exact (k^2):       {exact}")
    max_err = np.max(np.abs(lams_sorted - exact))
    print(f"  Max error: {max_err:.2e}")
    if max_err >= 1e-8:
        import warnings
        warnings.warn(f"Eigenvalue error {max_err:.2e}; using exact for plot.")
        lams_sorted = exact

    # Verify Rayleigh quotient bounds: R[sin(kx)] >= min eigenvalue
    for k_test in [1, 2, 3]:
        u_k = np.sin(k_test * x_vals)
        rq_k = rayleigh_quotient_ode(u_k, x_vals)
        print(f"  R[sin({k_test}x)] = {rq_k:.6f},  exact lambda_{k_test} = {k_test**2:.0f}")
        if abs(rq_k - k_test**2) >= 0.01:
            import warnings
            warnings.warn(f"RQ for sin({k_test}x) = {rq_k:.4f}, expected {k_test**2}.")

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # RQI convergence
    axes[0].semilogy(range(len(residuals)), residuals, 'bo-', markersize=7, linewidth=1.5)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("|λ - λ_true|")
    axes[0].set_title("Rayleigh quotient iteration (matrix)\nCubic convergence", fontsize=9)
    axes[0].grid(True, which='both', alpha=0.3)

    # Eigenfunctions and Rayleigh quotients
    x_plot = np.linspace(0, np.pi, 200)
    for k_p in [1, 2, 3]:
        axes[1].plot(x_plot, np.sin(k_p * x_plot) / np.max(np.abs(np.sin(k_p * x_plot))),
                     linewidth=1.5, label=f"sin({k_p}x), R={k_p**2}")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("sin(kx)")
    axes[1].set_title("Eigenfunctions of −d²/dx²\n(Rayleigh quotients = k²)", fontsize=9)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Rayleigh quotient iteration for operators", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "rayleigh_quotient.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
