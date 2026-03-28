"""Abscissa of the linearized Vlasov-Poisson operator.

Computes the numerical abscissa of the Volterra integral operator
arising in the linearization of the Vlasov-Poisson plasma equation:
  K(s,t) = (1 - a^2*(t-s)^2) * exp(-a^2*(t-s)^2/2)

following integro/VlasovPoisson.m by Toby Driscoll (October 2010).

The numerical abscissa is the maximum eigenvalue of B = (A + A^T)/2,
where A is the Volterra operator.

Original MATLAB: https://www.chebfun.org/examples/integro/VlasovPoisson.html
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chebfunjax.plotting import chebfun_style
chebfun_style()

import numpy as np
from scipy.linalg import eigh
import os

def run():
    print("=" * 60)
    print("Numerical abscissa of Vlasov-Poisson operator")
    print("=" * 60)

    # Domain [0, 6]
    # K(s,t) = (1 - a^2*(t-s)^2) * exp(-a^2*(t-s)^2/2)
    # A is Volterra: (Au)(t) = int_0^t K(s,t) u(s) ds
    # B = (A + A^T)/2 = Fredholm with same kernel (symmetric)
    # Numerical abscissa = max eigenvalue of B

    N = 100  # quadrature points
    T_dom = 6.0

    # Gauss-Legendre quadrature on [0, T_dom]
    from numpy.polynomial.legendre import leggauss
    t_gl_ref, w_ref = leggauss(N)
    # Map to [0, T_dom]
    t_gl = (T_dom / 2) * (t_gl_ref + 1)
    w_gl = (T_dom / 2) * w_ref

    def numerical_abscissa(a):
        """Compute max eigenvalue of B = Fredholm op with kernel K."""
        K_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                diff = t_gl[i] - t_gl[j]
                K_mat[i, j] = (1 - a**2 * diff**2) * np.exp(-0.5 * a**2 * diff**2)

        # B = (A + A^T)/2 = 0.5 * Fredholm with K_mat symmetric
        B_mat = 0.5 * K_mat

        # Discretize as matrix: (B u)(t_i) = sum_j K(t_i,t_j) u(t_j) * w_j * 0.5
        B_disc = B_mat * w_gl[np.newaxis, :]

        # Max eigenvalue of B_disc
        evals = np.linalg.eigvalsh(0.5 * (B_disc + B_disc.T))
        return np.max(evals)

    # Check at a=0: K=1 so the symmetrized operator has max eigenvalue ≈ T/2 ≈ 3.0
    # (the exact value is T/pi * something, numerically ~3.16 for T=6)
    na_a0 = numerical_abscissa(0.0)
    print(f"\na=0: numerical abscissa = {na_a0:.4f} (expected near T/2 = {T_dom/2:.4f})")
    assert 2.5 < na_a0 < 4.0, f"a=0 result out of range: {na_a0}"

    # Scan over a ∈ [0, 0.75]
    print("\nComputing numerical abscissa for a ∈ [0, 0.75]...")
    a_vals = np.linspace(0, 0.75, 40)
    omega_vals = np.array([numerical_abscissa(a) for a in a_vals])

    print(f"  Range of omega: [{omega_vals.min():.4f}, {omega_vals.max():.4f}]")

    # Find breakpoint (non-smooth transition)
    d_omega = np.gradient(omega_vals, a_vals)
    d2_omega = np.gradient(d_omega, a_vals)
    breakpoint_idx = np.argmax(np.abs(d2_omega[2:-2])) + 2
    a_star = a_vals[breakpoint_idx]
    print(f"  Estimated breakpoint at a* ≈ {a_star:.3f}")
    print(f"  (Chebfun finds a* ≈ 0.46 due to eigenvalue crossing)")

    # At a=0 the formula is exact
    assert omega_vals[0] > 2.0, "omega(0) should be positive"

    # --- Plot ---
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(a_vals, omega_vals, color='#0072BD', linestyle='-', linewidth=2)
    axes[0].axvline(a_star, color='#D95319', linestyle='--', alpha=0.7,
                    label=f'a* ≈ {a_star:.2f}')
    axes[0].set_title("Numerical abscissa ω(a) for Vlasov-Poisson", fontsize=11)
    axes[0].legend()

    # Kernel visualization at a=0.3
    a_plot = 0.3
    K_plot = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            diff = t_gl[i] - t_gl[j]
            K_plot[i, j] = (1 - a_plot**2 * diff**2) * np.exp(-0.5 * a_plot**2 * diff**2)

    im = axes[1].contourf(t_gl, t_gl, K_plot, levels=20, cmap='RdBu_r')
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title(f"Kernel K(s,t) for a={a_plot}", fontsize=11)

    fig.suptitle("Vlasov-Poisson operator numerical abscissa", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "vlasov_poisson.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll checks passed.")
    return True

if __name__ == "__main__":
    run()
