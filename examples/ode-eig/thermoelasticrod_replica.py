"""Stability of a thermoelastic rod.

Faithful replica of ode-eig/ThermoelasticRod.m by Toby Driscoll
(November 2011): the eigenvalue problem

    phi'' = lam phi,  0 < x < 1,
    phi(0) = 0,  phi'(1) + phi(1) = 4 delta int_0^1 phi dx,

whose Barber-condition integral term is just another linear boundary
condition from the Chebfun point of view.  The stability transition at
delta = 1 is recovered by chebfun rootfinding on the maximum
eigenvalue as a function of delta.

Original: https://www.chebfun.org/examples/ode-eig/ThermoelasticRod.html
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

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-eig')


def _solve(delta, k=4):
    N = Chebop(lambda x, u: u.diff(2), domain=(0, 1))
    N.lbc = 0.0                                    # fixed end
    # Barber condition: phi'(1) + phi(1) = 4 delta int phi
    N.bc = lambda x, u, d=delta: (u.diff()(1.0) + u(1.0)
                                  - 4 * d * u.sum())
    lam, V = N.eigs(k=k, sigma=0, return_eigenfunctions=True)
    lam = np.asarray(lam).real
    idx = np.argsort(lam)
    return lam[idx], [V[i] for i in idx]


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Stable case: all eigenvalues negative.
    Ls, Vs = _solve(0.96)
    print("ans =")
    for v in Ls:
        print(f"  {v/100:.15f}  (x 1e2)")

    # Slightly unstable case.
    Lu, Vu = _solve(1.02)
    print("ans =")
    for v in Lu:
        print(f"  {v/100:.15f}  (x 1e2)")

    # Least stable / unstable perturbation.
    xx = np.linspace(0, 1, 1000)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for ax, (lam, V, ttl) in zip(
            axes, [(Ls, Vs, "Stable"), (Lu, Vu, "Unstable")]):
        ax.plot(xx, np.asarray(V[3](xx)), lw=1.6)
        ax.set_title(f"{ttl}, $\\lambda$ = {lam[3]:.3f}")
        ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ThermoelasticRod_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Locate the stability transition by chebfun rootfinding.
    def maxlam(d):
        lam, _ = _solve(float(d), k=1)
        return float(lam[-1])

    stability = chebfun(
        lambda dd: np.array([maxlam(t) for t in np.atleast_1d(dd)],
                            dtype=np.float64),
        domain=(0.5, 2.0), eps=1e-11)
    print("stability =")
    print(stability)
    dstar = float(np.asarray((stability).roots())[0])
    print("dstar =")
    print(f"   {dstar:.15f}")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xx = np.linspace(0.5, 2.0, 1000)
    ax.plot(xx, np.asarray(stability(xx)), lw=1.6)
    ax.plot([dstar], [0.0], "ro", markersize=10)
    ax.set_xlabel("$\\delta$")
    ax.set_ylabel("max $\\lambda$")
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "ThermoelasticRod_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
