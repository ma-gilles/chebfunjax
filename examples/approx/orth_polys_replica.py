"""Orthogonal polynomials via Gram-Schmidt.

Faithful replica of approx/OrthPolys.m by Stefan Guettel (June 2011):
building orthonormal polynomials for an arbitrary weight by
Gram-Schmidt on chebfuns, and weighted least-squares approximation.

Original: https://www.chebfun.org/examples/approx/OrthPolys.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

XS = np.linspace(-1, 1, 2000)


def orth_poly(w, N):
    x = cj.chebfun(lambda t: t)
    P = [cj.chebfun(lambda t: 1.0 / np.sqrt(float(w.sum())) + 0 * t)]
    for k in range(N):
        xk = x * P[k]
        pk1 = xk
        for j in range(k + 1):
            C = float((w * xk * P[j]).sum())
            pk1 = pk1 - C * P[j]
        pk1 = pk1 * (1.0 / np.sqrt(float((w * pk1**2).sum())))
        P.append(pk1)
    return P


def run():
    os.makedirs(_IMG, exist_ok=True)

    w = cj.chebfun(lambda t: jnp.exp(jnp.pi * t))
    N = 5
    P = orth_poly(w, N)

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    for p in P:
        ax.plot(XS, np.asarray(p(jnp.asarray(XS))), lw=1.6)
    ax.set_title("Orthogonal polynomials on [-1,1] wrt w = exp(pi*x)",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OrthPolys_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Verify orthonormality
    I = np.array([[float((w * pi_ * pj_).sum()) for pj_ in P]
                  for pi_ in P])
    err = np.linalg.norm(I - np.eye(N + 1))
    print("err =")
    print(f"     {err:.15e}")

    # Weighted least-squares approximation of |x|
    f = cj.chebfun(lambda t: jnp.abs(t), domain=[-1.0, 0.0, 1.0])
    alpha = [float((w * p * f).sum()) for p in P]
    pstar_vals = sum(a * np.asarray(p(jnp.asarray(XS)))
                     for a, p in zip(alpha, P))
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(XS, np.abs(XS), 'b', lw=1.6)
    ax.plot(XS, pstar_vals, '--r', lw=1.6)
    ax.set_title("Least-squares approximation to |x| wrt w = exp(pi*x)",
                 fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OrthPolys_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
