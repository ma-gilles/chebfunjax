"""Rayleigh quotient iteration for an operator.

Faithful replica of ode-eig/RayleighQuotient.m by Nick Hale and Yuji
Nakatsukasa (March 2017, revised July 2019): RQI for symmetric and
nonsymmetric matrices (cubic vs quadratic convergence, restored to
cubic by the two-sided iteration), then the same code pattern for
selfadjoint and non-selfadjoint chebops, with `A - lam*I` solves and
the adjoint playing the role of the conjugate transpose.

All random data (matrices, initial vectors, and the randnfun initial
guesses sampled at chebpts(257)) is dumped from MATLAB's rng(10)
stream -- see _rayleighquotient_data.py.

Original: https://www.chebfun.org/examples/ode-eig/RayleighQuotient.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax.numpy as jnp
from _rayleighquotient_data import (
    A1,
    A1_SHAPE,
    A2,
    A2_SHAPE,
    A3,
    A3_SHAPE,
    F1,
    F2,
    F3,
    F4,
    U1,
    U2I,
    U2R,
    U3I,
    U3R,
    V3,
)

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
from chebfunjax.operators.adjoint import adjoint
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')

TOL = 1e-10


def _fmt(z):
    if abs(np.imag(z)) < 5e-16:
        return f"   {np.real(z):.15f}"
    return f"  {np.real(z):.15f} {np.imag(z):+.15f}i"


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # ---------------- 1. Symmetric matrix ----------------
    A = np.array(A1).reshape(A1_SHAPE)
    identity = np.eye(10)
    print("lam:")
    lam = A[-1, -1]
    print(_fmt(lam))
    u = np.array(U1)
    u = u / np.linalg.norm(u)
    res = [np.linalg.norm(A @ u - lam * u) / np.linalg.norm(A @ u)]
    while res[-1] > TOL:
        u = np.linalg.solve(A - lam * identity, u)
        u = u / np.linalg.norm(u)
        lam = u.conj() @ A @ u
        print(_fmt(lam))
        res.append(np.linalg.norm(A @ u - lam * u) / np.linalg.norm(A @ u))
    print("res =")
    for r in res:
        print(f"   {r:.15f}")

    # ---------------- 2. Nonsymmetric matrix ----------------
    A = np.array(A2).reshape(A2_SHAPE)
    print("lam:")
    lam = A[-1, -1]
    print(_fmt(lam))
    u = np.array(U2R) + 1j * np.array(U2I)
    u = u / np.linalg.norm(u)
    res2 = [np.linalg.norm(A @ u - lam * u) / np.linalg.norm(A @ u)]
    while res2[-1] > TOL:
        u = np.linalg.solve(A - lam * identity, u)
        u = u / np.linalg.norm(u)
        lam = u.conj() @ A @ u
        print(_fmt(lam))
        res2.append(np.linalg.norm(A @ u - lam * u) / np.linalg.norm(A @ u))
    print("res2 =")
    for r in res2:
        print(f"   {r:.15f}")

    # Two-sided RQI restores cubic convergence.
    A = np.array(A3).reshape(A3_SHAPE)
    print("lam:")
    lam = A[-1, -1]
    print(_fmt(lam))
    u = np.array(U3R) + 1j * np.array(U3I)
    u = u / np.linalg.norm(u)
    v = np.array(V3)
    v = v / np.linalg.norm(v)
    res3 = [np.linalg.norm(A @ u - lam * u) / np.linalg.norm(A @ u)]
    while res3[-1] > TOL:
        u = np.linalg.solve(A - lam * identity, u)
        u = u / np.linalg.norm(u)
        v = np.linalg.solve(A.conj().T - lam * identity, v)
        v = v / np.linalg.norm(v)
        lam = (v.conj() @ A @ u) / (v.conj() @ u)
        print(_fmt(lam))
        res3.append(np.linalg.norm(A @ u - lam * u) / np.linalg.norm(A @ u))
    print("res3 =")
    for r in res3:
        print(f"   {r:.15f}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.semilogy(range(len(res)), res, 'b-o')
    ax.text(len(res) - 0.9, res[-1] * 3, 'symm', color='b')
    ax.semilogy(range(len(res2)), res2, 'r--x')
    ax.text(len(res2) - 1.9, res2[-2], 'nonsymm', color='r')
    ax.semilogy(range(len(res3)), res3, 'm--^')
    ax.text(len(res3) - 1.9, res3[-2], 'nonsymm two-sided', color='m')
    ax.set_xlabel('iteration')
    ax.set_ylabel('residual')
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "RayleighQuotient_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 3. Selfadjoint linear operator ----------------
    dom = (-np.pi / 2, np.pi / 2)
    ddom = Domain(dom)

    def chebop_shifted(op_fn, lam):
        N = Chebop(lambda x, u: op_fn(x, u) - lam * u, domain=dom)
        N.lbc = 0.0
        N.rbc = 0.0
        return N

    def opA(x, u):
        return -u.diff(2)

    A_op = Chebop(lambda x, u: opA(x, u), domain=dom)
    A_op.lbc = 0.0
    A_op.rbc = 0.0

    def rqi(op_fn, u0, lam):
        print("lam:")
        print(_fmt(lam))
        u = u0 * (1.0 / float(u0.norm(2)))
        Au = op_fn(None, u)
        res = [float((Au - lam * u).norm(2)) / float(Au.norm(2))]
        while res[-1] > TOL:
            u = chebop_shifted(op_fn, lam).solve(u)
            u = u * (1.0 / float(u.norm(2)))
            Au = op_fn(None, u)
            lam = float((u * Au).sum())
            print(_fmt(lam))
            res.append(float((Au - lam * u).norm(2)) / float(Au.norm(2)))
        return res

    def opA_fun(x, u):
        return -u.diff(2)

    u0 = Chebfun.from_values(jnp.asarray(np.array(F1)), ddom)
    res = rqi(opA_fun, u0, 3.8)
    print("res =")
    for r in res:
        print(f"   {r:.15f}")

    # ---------------- 4. Non-selfadjoint operator ----------------
    def opB_fun(x, u):
        return -u.diff(2) + u.diff() + u

    u0 = Chebfun.from_values(jnp.asarray(np.array(F2)), ddom)
    res2 = rqi(opB_fun, u0, 1.0)
    print("res2 =")
    for r in res2:
        print(f"   {r:.15f}")

    # Two-sided iteration with the adjoint.
    B_op = Chebop(lambda x, u: opB_fun(x, u), domain=dom)
    B_op.lbc = 0.0
    B_op.rbc = 0.0
    Badj = adjoint(B_op)

    def opBadj_fun(x, u):
        return Badj.op(x, u) if callable(Badj.op) else None

    def chebop_shifted_adj(lam):
        N = Chebop(lambda x, u: Badj.op(x, u) - lam * u, domain=dom)
        N.lbc = 0.0
        N.rbc = 0.0
        return N

    print("lam:")
    lam = 1.0
    print(_fmt(lam))
    u = Chebfun.from_values(jnp.asarray(np.array(F3)), ddom)
    u = u * (1.0 / float(u.norm(2)))
    v = Chebfun.from_values(jnp.asarray(np.array(F4)), ddom)
    v = v * (1.0 / float(v.norm(2)))
    Au = opB_fun(None, u)
    res3 = [float((Au - lam * u).norm(2)) / float(Au.norm(2))]
    while res3[-1] > TOL:
        u = chebop_shifted(opB_fun, lam).solve(u)
        u = u * (1.0 / float(u.norm(2)))
        v = chebop_shifted_adj(lam).solve(v)
        v = v * (1.0 / float(v.norm(2)))
        Au = opB_fun(None, u)
        lam = float((v * Au).sum()) / float((v * u).sum())
        print(_fmt(lam))
        res3.append(float((Au - lam * u).norm(2)) / float(Au.norm(2)))
    print("res3 =")
    for r in res3:
        print(f"   {r:.15f}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.semilogy(range(len(res)), res, 'b-o')
    ax.text(len(res) - 0.9, res[-1] * 3, 'selfadj', color='b')
    ax.semilogy(range(len(res2)), res2, 'r--x')
    ax.text(len(res2) - 1.9, res2[-2], 'non-selfadj', color='r')
    ax.semilogy(range(len(res3)), res3, 'm--^')
    ax.text(len(res3) - 1.9, res3[-2], 'non-selfadj two-sided', color='m')
    ax.set_xlabel('iteration')
    ax.set_ylabel('residual')
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "RayleighQuotient_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
