"""A parameter dependent ODE with breakpoints.

Faithful replica of ode-linear/ParameterODE.m by Asgeir Birkisson
(January 2012): the BVP

    (a(x,s) u')' = 1,   u(0) = u(1) = 0,   a(x,s) = 1 + 4s(x^2 - x),

whose exact solution log(a(x,s))/(8s) develops a singularity at
x = 1/2 as s -> 1.  Solving for s = 1 - 10^{-gamma} shows accuracy
degrading on a plain [0,1] domain but restored by placing a
breakpoint at x = 1/2.

Original: https://www.chebfun.org/examples/ode-linear/ParameterODE.html
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

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
AX = [0, 1, -2.2, 0.2]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ParameterODE_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _a(x, s):
    return 1 + 4 * s * (x**2 - x)


def _ap(x, s):
    return 4 * s * (2 * x - 1)


def _uexact(x, s):
    return np.log(_a(x, s)) / (8 * s)


def _solve(s, dom):
    N = Chebop(lambda x, u: _a(x, s) * u.diff(2) + _ap(x, s) * u.diff(),
               domain=dom)
    N.lbc = 0
    N.rbc = 0
    return N.solve(1.0)


def _resid_err(u, s, dom):
    xf = Chebfun.identity(Domain(tuple(float(v) for v in dom)))
    res = float((_a(xf, s) * u.diff(2) + _ap(xf, s) * u.diff()
                 - 1).norm())
    t = np.linspace(1e-6, 1 - 1e-6, 2001)
    err = float(np.max(np.abs(np.asarray(u(jnp.asarray(t)))
                              - _uexact(t, s))))
    return res, err


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    tt = np.linspace(1e-6, 1 - 1e-6, 1500)

    # Plain domain [0, 1], gamma = 1..3
    res, err = [], []
    for g in (1, 2, 3):
        s = 1 - 10.0**(-g)
        u = _solve(s, (0, 1))
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.plot(tt, np.asarray(u(jnp.asarray(tt))), lw=1.6)
        ax.axis(AX)
        ax.grid(True)
        ax.set_title(f"gamma = {g}    length(solution) = {len(u):4d}",
                     fontsize=12)
        _save(fig)
        r, e = _resid_err(u, s, (0, 1))
        res.append(r)
        err.append(e)
        print(f"gamma = {g}: residual = {r:.3e}   error = {e:.3e}")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.semilogy(range(1, 4), res, '-*m', lw=1.6)
    ax.grid(True)
    ax.set_title("Norm of residual", fontsize=12)
    ax.set_xlabel(r"$\gamma$", fontsize=12)
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.semilogy(range(1, 4), err, '-*r', lw=1.6)
    ax.grid(True)
    ax.set_title("Norm of error", fontsize=12)
    ax.set_xlabel(r"$\gamma$", fontsize=12)
    _save(fig)

    # Breakpoint domain [0, 0.5, 1], gamma = 1..7
    err = []
    for g in range(1, 8):
        s = 1 - 10.0**(-g)
        u = _solve(s, (0, 0.5, 1))
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        ax.plot(tt, np.asarray(u(jnp.asarray(tt))), lw=1.6)
        ax.axis(AX)
        ax.grid(True)
        ax.set_title(f"gamma = {g}    length(solution) = {len(u):4d}",
                     fontsize=12)
        _save(fig)
        _, e = _resid_err(u, s, (0, 0.5, 1))
        err.append(e)
        print(f"gamma = {g}: error = {e:.3e}   length = {len(u)}")

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.semilogy(range(1, 8), err, '-*r', lw=1.6)
    ax.grid(True)
    ax.set_title("Norm of error", fontsize=12)
    ax.set_xlabel(r"$\gamma$", fontsize=12)
    _save(fig)


if __name__ == "__main__":
    run()
