"""The nonlinear optical response of a simple molecule.

Faithful replica of ode-eig/OpticalResponse.m by Jared L. Aurentz and
John S. Minor (September 2014): the polarization of an electron bound
by the quadratic potential V = 2x^2 under an applied field E,

    H(E) = -1/2 d^2/dx^2 + 2x^2 + E x,
    P(E) = <psi_1 | x | psi_1> / <psi_1 | psi_1>,

built as a chebfun in E and differentiated at 0 to read off the
optical response coefficients (alpha = 1/4, beta = 0, gamma ~ 0).

Original: https://www.chebfun.org/examples/ode-eig/OpticalResponse.html
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

L = 8.0


def _H(E):
    N = Chebop(lambda x, u: -0.5 * u.diff(2) + 2 * x**2 * u + E * x * u,
               domain=(-L, L))
    N.bc = "dirichlet"
    return N


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # Hermite functions: the four lowest eigenstates at E = 0.
    lam, PSI = _H(0.0).eigs(k=4, sigma="SR", return_eigenfunctions=True)
    xx = np.linspace(-L, L, 2000)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for psi in PSI:
        ax.plot(xx, np.asarray(psi(xx)), lw=2)
    ax.set_title("Hermite Functions", fontsize=16)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OpticalResponse_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Polarization as a chebfun in the field strength.
    Emax = 0.1
    x = chebfun(lambda t: t, domain=(-L, L))

    def polarization(efield):
        _, psis = _H(float(efield)).eigs(
            k=1, sigma="SR", return_eigenfunctions=True)
        psi = psis[0]
        return float(((x * psi) * psi).sum() / (psi * psi).sum())

    P = chebfun(
        lambda ee: np.array([polarization(t) for t in np.atleast_1d(ee)],
                            dtype=np.float64),
        domain=(-Emax, Emax), eps=1e-10)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ee = np.linspace(-Emax, Emax, 1000)
    ax.plot(ee, np.asarray(P(ee)), lw=2)
    ax.set_title("Polarization v. Electric Field", fontsize=16)
    ax.set_xlabel("Electric Field", fontsize=16)
    ax.set_ylabel("Polarization", fontsize=16)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "OpticalResponse_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Optical response coefficients from derivatives at E = 0.
    dP = P.diff()
    print("alpha =")
    print(f"   {float(dP(0.0)):.15f}")

    d2P = P.diff(2)
    print("beta =")
    print(f"   {float(d2P(0.0)) / 2:.15f}")

    d3P = P.diff(3)
    print("gamma =")
    print(f"   {float(d3P(0.0)) / 6:.15f}")


if __name__ == "__main__":
    run()
