"""Floquet theory of periodic ODEs.

Faithful replica of ode-linear/Floquet.m by Marcus Webb (January
2015): the fundamental matrix of a coupled Mathieu system over one
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
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

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


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    Phi = [list(_solve(u)) for u in range(4)]   # Phi[col][row]
    n = 4
    PhiT = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            PhiT[i, j] = float(Phi[j][i](jnp.asarray(T)))

    B = logm(PhiT) / T
    lam, V = np.linalg.eig(B)
    order = np.lexsort((lam.real, np.round(np.abs(lam.imag), 6)))
    lam, V = lam[order], V[:, order]
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

    # Periodic factor P(t) = Phi(t) exp(-tB)
    tt = np.linspace(0, T, 300)
    PhiV = np.zeros((n, n, tt.size))
    for i in range(n):
        for j in range(n):
            PhiV[i, j] = np.asarray(Phi[j][i](jnp.asarray(tt)))
    Pt = np.zeros((n, n, tt.size), dtype=complex)
    for k, tk in enumerate(tt):
        E = V @ np.diag(np.exp(-tk * lam)) @ invV
        Pt[:, :, k] = PhiV[:, :, k] @ E

    fig, axs = plt.subplots(n, n, figsize=(11.5, 9.0))
    for i in range(n):
        for j in range(n):
            axs[i, j].plot(tt, Pt[i, j].real, lw=1.6)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    fig.suptitle("Entries of the periodic matrix P(i,j)(t)",
                 fontsize=13)
    fig.set_facecolor("white")
    fig.savefig(os.path.join(_IMG, "Floquet_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Periodicity check: P(0) vs P(T)
    err = float(np.max(np.abs(Pt[:, :, 0] - Pt[:, :, -1])))
    print(f"max |P(0) - P(T)| = {err:.2e}")


if __name__ == "__main__":
    run()
