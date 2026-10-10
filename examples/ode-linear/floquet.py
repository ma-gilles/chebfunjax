"""Floquet theory of periodic ODEs.

Translation of ode-linear/Floquet.m by Richard Mikael Slevinsky (October
2014): the fundamental matrix of a coupled Mathieu system over one
period, the monodromy matrix and its logarithm, Floquet exponents and
multipliers, and the periodic factor P(t) of the Floquet
decomposition Phi(t) = P(t) exp(tB).

Original: https://www.chebfun.org/examples/ode-linear/Floquet.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax import chebfun
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

T = np.pi
ALPHA = 0.15


def _solve(unit):
    A = Chebop(lambda t, x1, x2, y1, y2: [
        x1.diff() - x2,
        x2.diff() - y1 + (2 + ALPHA * (2 * t).cos()) * x1,
        y1.diff() - y2,
        y2.diff() - x1 + (2 + ALPHA * (2 * t).cos()) * y1],
        domain=(0, T))
    A.lbc = (lambda x1, x2, y1, y2, _u=unit:
             [v - 1 if i == _u else v
              for i, v in enumerate((x1, x2, y1, y2))])
    return A.solve(0.0)


def _exponential_matrix(eigenvalues, vectors, inverse_vectors, domain, sign):
    """Source diagonal Chebfun exponential and two public matrix products."""
    t = chebfun(lambda t: t, domain=domain)
    n = len(eigenvalues)
    diagonal = ChebMatrix([
        [(sign*t*complex(eigenvalues[i])).exp() if i == j else 0
         for j in range(n)] for i in range(n)], domain=domain)
    left = ChebMatrix(vectors.tolist(), domain=domain)
    right = ChebMatrix(inverse_vectors.tolist(), domain=domain)
    return left * diagonal * right


def run(initial_conditions=None):
    os.makedirs(_IMG, exist_ok=True)

    Phi = [list(_solve(u)) for u in range(4)]   # Phi[col][row]
    n = 4
    PhiT = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            PhiT[i, j] = float(Phi[j][i](jnp.asarray(T)))

    B = logm(PhiT) / T
    lam, V = np.linalg.eig(B)
    invV = np.linalg.inv(V)

    print("Exponents =")
    for v in lam:
        print(f"  {v.real:.15f} {'-' if v.imag < 0 else '+'} "
              f"{abs(v.imag):.15f}i")
    mult = np.exp(lam * T)
    print("Multipliers =")
    for v in mult:
        print(f"  {v.real:.15f} {'-' if v.imag < 0 else '+'} "
              f"{abs(v.imag):.15f}i")

    # Source matrix operations construct continuous Chebfuns, then each
    # product entry is rebuilt with the original periodic flag.
    expmB = _exponential_matrix(lam, V, invV, (0.0, T), -1)
    periodic = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            temp = Phi[0][i] * expmB.blocks[0][j]
            for k in range(1, n):
                temp = temp + Phi[k][i] * expmB.blocks[k][j]
            periodic[i][j] = chebfun(
                lambda t, temp=temp: temp(t), domain=(0.0, T), periodic=True)

    fig, axs = plt.subplots(n, n, figsize=(6, 2.7), dpi=100)
    for i in range(n):
        for j in range(n):
            matlab_plot(periodic[i][j].real(), ax=axs[i, j], linewidth=2.0)
            axs[i, j].tick_params(labelsize=8, pad=1)
    fig.suptitle("Entries of the periodic matrix P(i,j)(t)",
                 fontsize=9, fontweight="bold", y=.99)
    fig.subplots_adjust(left=.145, right=.905, bottom=.13, top=.92,
                        wspace=.45, hspace=.65)
    _savefig(fig, os.path.join(_IMG, "Floquet_01.png"), size=(600, 270))
    plt.close(fig)

    # Preserve the original ten-period construction and source summation order.
    domain = (0.0, 10*T)
    expmB = _exponential_matrix(lam, V, invV, domain, 1)
    if initial_conditions is None:
        # Explicit reproducible RNG adapter. Native rand has an ambient stream;
        # this draw does not claim to reproduce that stream or the cached figure.
        x0 = jax.random.uniform(jax.random.PRNGKey(0), (n,), dtype=jnp.float64)
    else:
        x0 = jnp.asarray(initial_conditions, dtype=jnp.float64)
        if x0.shape != (n,):
            raise ValueError("initial_conditions must have shape (4,)")
    temp = [expmB.blocks[i][0] * x0[0] for i in range(n)]
    for i in range(n):
        for j in range(1, n):
            temp[i] = temp[i] + expmB.blocks[i][j] * x0[j]
    solution = [chebfun(lambda t, entry=periodic[i][0]: entry(t),
                        domain=domain) * temp[0] for i in range(n)]
    for i in range(n):
        for j in range(1, n):
            entry = periodic[i][j]
            continued = chebfun(lambda t, entry=entry: entry(t), domain=domain)
            solution[i] = solution[i] + continued * temp[j]

    fig, ax = plt.subplots(figsize=(6, 2.7), dpi=100)
    for component in solution:
        matlab_plot(component.real(), ax=ax, linewidth=2.0)
    ax.set_xlabel('t', fontsize=9)
    ax.set_ylabel('x(t) and y(t)', fontsize=9)
    ax.set_title('Solution of the system of coupled oscillators with periodic parametric excitation',
                 fontsize=9, fontweight='bold')
    ax.legend(['x(t)', "x'(t)", 'y(t)', "y'(t)"],
              fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.13, right=.905, bottom=.16, top=.90)
    _savefig(fig, os.path.join(_IMG, "Floquet_02.png"), size=(600, 270))
    plt.close(fig)
    return {"Phi": Phi, "B": B, "exponents": lam, "periodic": periodic,
            "initial_conditions": x0, "solution": solution}


if __name__ == "__main__":
    run()
