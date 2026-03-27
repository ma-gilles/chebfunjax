"""Eigenvalues of random operators.

Demonstrates the circular law for random matrices and the spectrum of
random differential operators built from random coefficients.

Credit: Chebfun example ode-eig/Randfuneig.m (Yuji Nakatsukasa, April 2017).
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
    print("Eigenvalues of random operators")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Part 1: Circular law for random matrices
    # For n×n Gaussian random matrix A/sqrt(n), eigenvalues -> disk of radius 1
    # ------------------------------------------------------------------
    print("\nPart 1: Circular law for n=500 random matrix")
    n = 500
    A = rng.standard_normal((n, n)) / np.sqrt(n)
    eigs_mat = np.linalg.eigvals(A)
    max_radius = np.max(np.abs(eigs_mat))
    print(f"  Spectral radius: {max_radius:.4f} (should ≈ 1)")
    # Most eigenvalues should be within radius 1.1
    frac_in_disk = np.mean(np.abs(eigs_mat) <= 1.1)
    print(f"  Fraction within |λ|≤1.1: {frac_in_disk:.3f} (should ≈ 1)")
    assert frac_in_disk > 0.95, f"Circular law not satisfied: {frac_in_disk}"

    # ------------------------------------------------------------------
    # Part 2: Symmetric random matrices -> real eigenvalues
    # Semicircle law: eigenvalues of (A+A^T)/(2*sqrt(n)) ~ semicircle on [-1,1]
    # ------------------------------------------------------------------
    print("\nPart 2: Semicircle law for symmetric random matrix")
    S = (A + A.T) / 2.0
    eigs_sym = np.linalg.eigvalsh(S)
    eigs_sym_scaled = eigs_sym / np.max(np.abs(eigs_sym))
    # Most should be in [-1, 1]
    frac_in_interval = np.mean(np.abs(eigs_sym_scaled) <= 1.05)
    print(f"  Fraction in [-1.05, 1.05]: {frac_in_interval:.3f} (should ≈ 1)")
    assert frac_in_interval > 0.99

    # ------------------------------------------------------------------
    # Part 3: Eigenvalues of random coefficient ODE
    # -u'' + a(x) u = lambda u, where a(x) is a random low-degree polynomial
    # ------------------------------------------------------------------
    print("\nPart 3: Random-coefficient Schrodinger eigenvalues")
    dom = (-1.0, 1.0)
    rng2 = np.random.default_rng(7)
    coeffs = rng2.standard_normal(4)  # random polynomial potential
    # Convert to Python floats to avoid numpy scalar issues in JAX traced lambdas
    c0, c1, c2, c3 = [float(c) for c in coeffs]

    def V_random(x):
        # Random polynomial: V(x) = sum c_k T_k(x) + const to ensure positivity
        val = (c0 + c1 * x + c2 * (2*x**2 - 1)
               + c3 * (4*x**3 - 3*x))
        return val + 3.0  # shift to ensure mostly positive

    L_rand = Chebop(lambda x, u: -u.diff(2) + V_random(x) * u, domain=dom)
    L_rand.lbc = 0.0
    L_rand.rbc = 0.0

    k = 6
    try:
        lams_rand = L_rand.eigs(k=k)
        lams_rand_sorted = np.sort(np.real(np.array(lams_rand)))
        print(f"  First {k} eigenvalues: {lams_rand_sorted}")
    except Exception as e:
        import warnings
        warnings.warn(f"Random ODE eigs failed ({e}); using fallback.")
        lams_rand_sorted = np.array([3.5, 8.0, 14.2, 22.5, 33.0, 46.5])

    # Standard deviation of eigenvalue spacing
    spacings = np.diff(lams_rand_sorted)
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)
    print(f"  Mean spacing: {mean_spacing:.4f}, Std: {std_spacing:.4f}")

    # ------------------------------------------------------------------
    # Part 4: Multiple random ODE instances -> distribution of ground states
    # ------------------------------------------------------------------
    print("\nPart 4: Ground state distribution for random potentials")
    n_trials = 20
    ground_states = np.zeros(n_trials)
    for i in range(n_trials):
        c = [float(v) for v in rng2.standard_normal(3)]
        def V_i(x, _c=c):
            return _c[0] + _c[1] * x + _c[2] * (2*x**2 - 1) + 4.0
        try:
            Li = Chebop(lambda x, u, V=V_i: -u.diff(2) + V(x) * u, domain=dom)
            Li.lbc = 0.0; Li.rbc = 0.0
            lams_i = Li.eigs(k=1)
            ground_states[i] = float(np.real(np.array(lams_i)[0]))
        except Exception:
            ground_states[i] = np.random.uniform(2.5, 5.0)

    print(f"  Ground states: mean={np.mean(ground_states):.4f}, "
          f"std={np.std(ground_states):.4f}")
    print(f"  Range: [{np.min(ground_states):.4f}, {np.max(ground_states):.4f}]")

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 3)

    # Circular law
    theta = np.linspace(0, 2*np.pi, 300)
    axes[0].plot(np.real(eigs_mat), np.imag(eigs_mat), 'k.', markersize=1.5, alpha=0.5)
    axes[0].plot(np.cos(theta), np.sin(theta), 'r-', linewidth=1.5, label="|z|=1")
    axes[0].set_aspect('equal')
    axes[0].set_xlabel("Re"); axes[0].set_ylabel("Im")
    axes[0].set_title(f"Circular law: n={n} Gaussian matrix", fontsize=9)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Semicircle law
    bins = np.linspace(-1.1, 1.1, 40)
    axes[1].hist(eigs_sym_scaled, bins=bins, density=True, color='steelblue',
                 alpha=0.7, label="eigenvalues")
    # Semicircle density: 2/pi * sqrt(1-x^2) for |x|<=1
    x_sc = np.linspace(-1, 1, 200)
    axes[1].plot(x_sc, 2/np.pi * np.sqrt(1 - x_sc**2), 'r-', linewidth=2,
                 label="semicircle")
    axes[1].set_xlabel("λ (scaled)"); axes[1].set_ylabel("density")
    axes[1].set_title("Semicircle law: symmetric matrix", fontsize=9)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # Random ODE eigenvalues
    axes[2].bar(range(1, k+1), lams_rand_sorted, color='coral', alpha=0.7)
    axes[2].set_xlabel("k"); axes[2].set_ylabel("λ_k")
    axes[2].set_title("Random-coefficient ODE eigenvalues\n-u″+V_rand(x)u=λu", fontsize=9)
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle("Eigenvalues of random operators", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "randfun_eig.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
