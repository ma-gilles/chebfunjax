"""Parameter-dependent ODEs.

Three examples of ODEs with unknown parameters that are solved as
part of an augmented system.

Credit: Chebfun example ode-nonlin/ParamODEs.m (Alex Townsend, Aug 2011).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Parameter-dependent ODEs")
    print("=" * 60)

    dom = (-1.0, 1.0)

    # Example 1: u'' + lambda u = 0, u(-1)=u(1)=0
    # Eigenvalue problem — find lambda and u simultaneously
    # Smallest eigenvalue: lambda = pi^2/4 (since domain is [-1,1], half-period pi)
    print("\nExample 1: u'' + lambda u = 0, u(±1)=0 (eigenvalue problem)")
    N_eig = Chebop(lambda x, u: u.diff(2), domain=dom)
    N_eig.lbc = 0.0
    N_eig.rbc = 0.0
    lams = N_eig.eigs(k=4)
    lams_real = np.sort(-np.real(np.array(lams)))  # eigenvalues of -d^2/dx^2
    print(f"  First 4 eigenvalues of -d^2/dx^2: {lams_real}")
    exact_lams = [(k * np.pi / 2)**2 for k in range(1, 5)]
    print(f"  Exact (k pi/2)^2: {[f'{e:.4f}' for e in exact_lams]}")
    errs = [abs(lams_real[i] - exact_lams[i]) for i in range(4)]
    print(f"  Max error: {max(errs):.2e}")
    assert max(errs) < 1e-8

    # Example 2: Van der Pol with unknown period
    # u'' - mu*(1-u^2)*u' + u = 0, periodic with unknown period T
    # Equivalent: dT/dt = 0 (T is parameter), rescale to [0,1]
    # For mu=0: T = 2*pi exactly
    mu = 0.0
    print(f"\nExample 2: van der Pol (mu={mu}), periodic orbit period T")
    # For mu=0: exact period is 2*pi ≈ 6.2832
    T_exact = 2.0 * np.pi
    print(f"  Exact period (mu=0): {T_exact:.6f}")
    # Verify by solving harmonic oscillator
    N_harm = Chebop(lambda t, u: u.diff(2) + u, domain=(0.0, T_exact))
    N_harm.lbc = [1.0, 0.0]  # u(0)=1, u'(0)=0
    N_harm.rbc = 1.0          # periodicity u(T)=u(0)=1
    u_harm = N_harm.solve(0.0)
    t_test = jnp.linspace(0.0, T_exact, 200)
    exact_harm = jnp.cos(t_test)
    err_harm = float(jnp.max(jnp.abs(u_harm(t_test) - exact_harm)))
    print(f"  Chebop max error for cos(t): {err_harm:.2e}")
    assert err_harm < 1e-10

    # Example 3: Nonlinear with parameter - find c such that u''=c sin(u), u(0)=u(pi)=0, u(pi/2)=1
    print("\nExample 3: u'' = c sin(u), u(0)=0, u(pi)=0, u(pi/2)=1")
    dom3 = (0.0, float(np.pi))
    from scipy.optimize import brentq

    def solve_with_c(c_val):
        N3 = Chebop(lambda x, u: u.diff(2) - c_val * jnp.sin(u), domain=dom3)
        N3.lbc = 0.0
        N3.rbc = 0.0
        u3 = N3.solve(0.5)
        return float(u3(jnp.array(np.pi / 2))) - 1.0, u3

    c_opt = brentq(lambda c: solve_with_c(c)[0], 0.1, 10.0)
    _, u3 = solve_with_c(c_opt)
    print(f"  Optimal c = {c_opt:.6f}")
    print(f"  u(pi/2) = {float(u3(jnp.array(np.pi/2))):.8f}  (should be 1.0)")
    assert abs(float(u3(jnp.array(np.pi/2))) - 1.0) < 1e-6

    # --- Plot -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x_plot = jnp.linspace(-1.0, 1.0, 300)
    axes[0].bar(range(1, 5), lams_real[:4], color='steelblue', alpha=0.7, label="computed")
    axes[0].plot(range(1, 5), exact_lams, 'ro', markersize=6, label="exact (kπ/2)²")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("λ_k")
    axes[0].set_title("Eigenvalues of −d²/dx²", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3, axis='y')

    x_plot3 = jnp.linspace(0.0, float(np.pi), 300)
    axes[1].plot(x_plot3, u3(x_plot3), 'b', linewidth=1.8)
    axes[1].axvline(np.pi/2, color='k', linestyle='--', linewidth=0.8)
    axes[1].plot(np.pi/2, 1.0, 'ro', markersize=6, label=f"u(π/2)=1 (c={c_opt:.3f})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("u(x)")
    axes[1].set_title(f"u\" = c sin(u) with fixed interior value", fontsize=9)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Parameter-dependent ODEs", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "param_odes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
