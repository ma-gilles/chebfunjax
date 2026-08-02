"""Crouzeix's conjecture.

Faithful replica of linalg/Crouzeix.m by Nick Trefethen
(August 2017): the Crouzeix ratio ||p(A)|| / max_{W(A)}|p| for
several matrices — conjectured to be at most 2, with equality for a
Jordan block and p(z) = z.

Original: https://www.chebfun.org/examples/linalg/Crouzeix.html
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')


def _fov_point(A, theta_arr):
    theta_arr = np.atleast_1d(np.asarray(theta_arr, dtype=float))
    out = np.empty(theta_arr.shape, dtype=complex)
    for i, th in enumerate(theta_arr.ravel()):
        r = np.exp(1j * th)
        H = (r * A + np.conj(r) * A.conj().T) / 2
        _, V = np.linalg.eigh(H)
        v = V[:, -1]
        out.ravel()[i] = (v.conj() @ A @ v) / (v.conj() @ v)
    return out.reshape(theta_arr.shape)


def _fov_chebfun(A, splitting=False):
    op = lambda t: jnp.asarray(_fov_point(A, np.asarray(t)))  # noqa: E731
    f = cj.chebfun(op, domain=(0.0, 2 * np.pi),
                   splitting=splitting)
    return f.merge() if splitting else f


def _grcar(n, k=3):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, max(0, i - 1):min(n, i + k + 1)] = 1.0
        if i > 0:
            A[i, i - 1] = -1.0
    return A


def _crouzeix_ratio(A, a, splitting=False):
    """||polyvalm(a, A)|| / max |polyval(a, fov(A))|."""
    pA = np.zeros_like(A, dtype=complex)
    for coef in a:
        pA = pA @ A + coef * np.eye(A.shape[0])
    F = _fov_chebfun(A, splitting=splitting)
    pf = None
    npow = len(a) - 1
    for k, coef in enumerate(a):
        term = F ** (npow - k) * complex(coef) if npow - k > 0 \
            else None
        if npow - k == 0:
            term = F * 0 + complex(coef)
        pf = term if pf is None else pf + term
    _, m = pf.abs().max()
    return np.linalg.norm(pA, 2) / float(m)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    A = np.exp(1.4j) * _grcar(30)
    F = _fov_chebfun(A)
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    w = np.linalg.eigvals(A)
    ax.plot(w.real, w.imag, '.k', ms=10)
    t = np.linspace(0, 2 * np.pi, 800)
    v = np.asarray(F(t))
    ax.plot(v.real, v.imag, 'm', lw=1.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Crouzeix_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    A = np.array([[0, 1], [0, 0]], dtype=float)
    a = np.array([1.0, 0.0])
    print("ans =")
    print(f"   {_crouzeix_ratio(A, a):.15f}")

    rs = np.random.RandomState(5489)  # rng('default')
    A = rs.randn(20, 20) / np.sqrt(20)
    a = rs.randn(5)
    print("ans =")
    print(f"   {_crouzeix_ratio(A, a):.15f}")

    B = np.diag(np.linalg.eigvals(A))
    print("ans =")
    print(f"   {_crouzeix_ratio(B, a, splitting=True):.15f}")


if __name__ == "__main__":
    run()
