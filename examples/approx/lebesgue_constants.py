"""Lebesgue functions and Lebesgue constants.

Compares Chebyshev interpolation nodes versus equispaced nodes, showing
the Runge phenomenon through Lebesgue constants.
Based on Chebfun example approx/LebesgueConst.m by Nick Trefethen (November 2010).

Original: https://www.chebfun.org/examples/approx/LebesgueConst.html
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

def lebesgue_function(nodes, n_eval=1000):
    """Compute the Lebesgue function for a set of interpolation nodes."""
    nodes = np.asarray(nodes)
    n = len(nodes)
    x = np.linspace(-1, 1, n_eval)
    # Compute the barycentric weights
    L = np.zeros(n_eval)
    for j in range(n_eval):
        # Lagrange basis functions at x[j]
        val = np.abs(np.prod([(x[j] - nodes[k]) / (nodes[i] - nodes[k])
                               for i in range(n)
                               for k in range(n) if k != i], axis=0)
                     if n > 1 else 1.0)
        # More efficient direct computation
        lj = np.ones(n)
        for m in range(n):
            for k in range(n):
                if k != m:
                    lj[m] *= (x[j] - nodes[k]) / (nodes[m] - nodes[k])
        L[j] = np.sum(np.abs(lj))
    return x, L

def lebesgue_constant_and_function(nodes):
    """Compute Lebesgue function and constant for interpolation nodes."""
    nodes = np.sort(np.asarray(nodes, dtype=np.float64))
    n = len(nodes)
    n_eval = 800
    x = np.linspace(-1, 1, n_eval)
    L = np.zeros(n_eval)
    for j in range(n_eval):
        lj = np.zeros(n)
        for m in range(n):
            num = 1.0
            den = 1.0
            for k in range(n):
                if k != m:
                    num *= (x[j] - nodes[k])
                    den *= (nodes[m] - nodes[k])
            if abs(den) > 1e-300:
                lj[m] = num / den
            else:
                lj[m] = 0.0
        L[j] = np.sum(np.abs(lj))
    return x, L, np.max(L)

def run():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../../docs/images/approx')
    os.makedirs(outdir, exist_ok=True)

    # --- Chebyshev vs equispaced nodes, n=10 -----------------------------
    n = 10
    # Chebyshev nodes of the second kind (Gauss-Chebyshev-Lobatto)
    cheb_nodes = -np.cos(np.pi * np.arange(n) / (n - 1))
    equi_nodes = np.linspace(-1, 1, n)

    x_c, L_c, Lambda_c = lebesgue_constant_and_function(cheb_nodes)
    x_e, L_e, Lambda_e = lebesgue_constant_and_function(equi_nodes)

    fig, axes = plt.subplots(2, 2)

    axes[0, 0].plot(x_c, L_c, 'b-', linewidth=1.5)
    axes[0, 0].set_title(f'10 Chebyshev nodes,  $\\Lambda$ = {Lambda_c:.2f}', fontsize=11)
    axes[0, 0].set_ylim(bottom=0)

    axes[0, 1].plot(x_e, L_e, 'r-', linewidth=1.5)
    axes[0, 1].set_title(f'10 equispaced nodes,  $\\Lambda$ = {Lambda_e:.2f}', fontsize=11)
    axes[0, 1].set_ylim(bottom=0)

    # n=30 on semilogy
    n2 = 30
    cheb_nodes2 = -np.cos(np.pi * np.arange(n2) / (n2 - 1))
    equi_nodes2 = np.linspace(-1, 1, n2)
    x_c2, L_c2, Lambda_c2 = lebesgue_constant_and_function(cheb_nodes2)
    x_e2, L_e2, Lambda_e2 = lebesgue_constant_and_function(equi_nodes2)

    axes[1, 0].semilogy(x_c2, np.maximum(L_c2, 1e-14), 'b-', linewidth=1.5)
    axes[1, 0].set_title(f'30 Chebyshev nodes,  $\\Lambda$ = {Lambda_c2:.2f}', fontsize=11)

    axes[1, 1].semilogy(x_e2, np.maximum(L_e2, 1e-14), 'r-', linewidth=1.5)
    axes[1, 1].set_title(f'30 equispaced nodes,  $\\Lambda$ ≈ {Lambda_e2:.2e}', fontsize=11)

    fig.suptitle('Lebesgue functions: Chebyshev vs. equispaced nodes', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'lebesgue_constants.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"10-point Chebyshev Lambda = {Lambda_c:.3f}")
    print(f"10-point equispaced Lambda = {Lambda_e:.3f}")
    print(f"30-point Chebyshev Lambda = {Lambda_c2:.3f}")
    print(f"30-point equispaced Lambda ≈ {Lambda_e2:.2e}")

    # Chebyshev Lebesgue constant grows as (2/pi)*log(n)
    nn = np.array([10, 20, 30, 40, 50])
    cheb_lambdas = []
    for n3 in nn:
        nodes3 = -np.cos(np.pi * np.arange(n3) / (n3 - 1))
        _, _, lam = lebesgue_constant_and_function(nodes3)
        cheb_lambdas.append(lam)
    cheb_lambdas = np.array(cheb_lambdas)
    theory = (2 / np.pi) * np.log(nn) + 0.9625
    print("Chebyshev Lambda vs (2/pi)*log(n) + 0.9625:")
    for n3, l, t in zip(nn, cheb_lambdas, theory):
        print(f"  n={n3:3d}:  Lambda={l:.4f},  theory={t:.4f}")

    print("lebesgue_constants: done")
    return True

if __name__ == "__main__":
    run()
