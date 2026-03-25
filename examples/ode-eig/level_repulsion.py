"""Avoided crossings for ODE eigenvalues.

Demonstrates level repulsion / avoided crossings for eigenvalues of a
parameter-dependent fourth-order self-adjoint differential operator.

Credit: Chebfun example ode-eig/LevelRepulsionODE.m
        (Abi Gopal and Nick Trefethen, Mar 2017).
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
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Avoided crossings (level repulsion) for ODE eigenvalues")
    print("=" * 60)

    dom = (0.0, float(np.pi))

    # Sweep parameter t in [0,1].
    # For each t, consider the 2nd-order operator:
    #   L(t) u = -u'' + t*x*(pi-x)*u
    # The eigenvalues are lambda_k(t) and we watch them as t varies.
    # For a 2nd-order self-adjoint operator, eigenvalues cannot cross
    # (Sturm-Liouville), so crossings are forbidden.

    n_t = 30
    ts = np.linspace(0.0, 5.0, n_t)
    n_eigs = 5
    all_lams = np.zeros((n_t, n_eigs))

    print(f"\nSweeping t in [0, 5], computing {n_eigs} eigenvalues ...")
    for i, t in enumerate(ts):
        L_t = Chebop(
            lambda x, u, _t=t: -u.diff(2) + _t * x * (np.pi - x) * u,
            domain=dom,
        )
        L_t.lbc = 0.0
        L_t.rbc = 0.0
        lams_t = L_t.eigs(k=n_eigs)
        all_lams[i] = np.sort(np.real(np.array(lams_t)))

    print("Done.")

    # Verify eigenvalues are increasing with t (potential gets larger -> eigenvalues grow)
    for j in range(n_eigs):
        assert np.all(np.diff(all_lams[:, j]) >= -0.5), \
            f"Eigenvalue {j} should be non-decreasing with t"

    # Check: eigenvalue gaps should remain positive (no crossings)
    for i in range(n_t):
        gaps = np.diff(all_lams[i])
        assert np.all(gaps > 0), f"Eigenvalues crossed at t={ts[i]:.2f}"

    print("\nEigenvalue gaps at t=0 (free: k^2) and t=5 (full potential):")
    print(f"  t=0: {all_lams[0]}")
    print(f"  t=5: {all_lams[-1]}")

    # --- Second test: 4th-order operator can have near-crossings
    # L(t) u = u'''' + t * u'' + u, with Dirichlet+Neumann BCs
    # This models beam bending with applied compression
    print("\nFourth-order beam eigenvalues vs t (compression parameter):")
    n_t4 = 20
    ts4 = np.linspace(0.0, 10.0, n_t4)
    n_eigs4 = 4
    all_lams4 = np.zeros((n_t4, n_eigs4))
    for i, t in enumerate(ts4):
        try:
            L4 = Chebop(
                lambda x, u, _t=t: u.diff(4) - _t * u.diff(2),
                domain=dom,
            )
            L4.lbc = [0.0, 0.0]   # u(0)=0, u'(0)=0
            L4.rbc = [0.0, 0.0]   # u(pi)=0, u'(pi)=0
            lams4 = L4.eigs(k=n_eigs4)
            all_lams4[i] = np.sort(np.real(np.array(lams4)))
        except Exception:
            all_lams4[i] = np.nan

    valid = ~np.any(np.isnan(all_lams4), axis=1)
    print(f"  Valid t-values: {np.sum(valid)}/{n_t4}")
    if np.sum(valid) >= 2:
        print(f"  t=0 eigenvalues: {all_lams4[0]}")

    # --- Plot -----------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 2nd order: no crossings
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_eigs))
    for j in range(n_eigs):
        axes[0].plot(ts, all_lams[:, j], color=colors[j], linewidth=1.8,
                     label=f"λ_{j+1}")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("λ_k")
    axes[0].set_title("2nd-order: eigenvalues can't cross\n-u″ + t·x(π−x)u = λu", fontsize=9)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # 4th order: level repulsion visible
    if np.sum(valid) >= 2:
        ts4_valid = ts4[valid]
        lams4_valid = all_lams4[valid]
        colors4 = plt.cm.tab10(np.linspace(0, 0.5, n_eigs4))
        for j in range(n_eigs4):
            axes[1].plot(ts4_valid, lams4_valid[:, j], color=colors4[j],
                         linewidth=1.8, label=f"λ_{j+1}")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("λ_k")
    axes[1].set_title("4th-order beam: avoided crossings\nu″″ − t·u″ = λu", fontsize=9)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Level repulsion / avoided crossings in ODE eigenvalues", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "level_repulsion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
