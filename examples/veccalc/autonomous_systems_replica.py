"""Phase portraits and trajectories.

Faithful replica of veccalc/AutonomousSystems.m by Alex Townsend,
March 2013: phase portraits of the simple harmonic oscillator, the
nonlinear pendulum, and the Duffing oscillator; trajectories are
integrated with RK45 (MATLAB ode45 defaults) over the chebfun2v field,
and the Duffing critical points are found with roots().

Original: https://www.chebfun.org/examples/veccalc/AutonomousSystems.html
Copyright 2013 by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'veccalc')


def _field_rhs(F):
    f1 = Chebfun2(approx=F.components[0])
    f2 = Chebfun2(approx=F.components[1])

    def rhs(t, y):
        return [float(np.asarray(f1(y[0], y[1]))),
                float(np.asarray(f2(y[0], y[1])))]
    return rhs


def _quiver(ax, F, dom, n=14):
    xq = np.linspace(dom[0], dom[1], n)
    yq = np.linspace(dom[2], dom[3], n)
    Xq, Yq = np.meshgrid(xq, yq)
    U = np.asarray(Chebfun2(approx=F.components[0])(
        jnp.asarray(Xq.ravel()), jnp.asarray(Yq.ravel()))).reshape(Xq.shape)
    V = np.asarray(Chebfun2(approx=F.components[1])(
        jnp.asarray(Xq.ravel()), jnp.asarray(Yq.ravel()))).reshape(Xq.shape)
    ax.quiver(Xq, Yq, U, V, color="b")


def run():
    os.makedirs(_IMG, exist_ok=True)

    # -- Simple harmonic oscillator ----------------------------------
    dom = (-1.0, 1.0, -3.0, 3.0)
    w = 2.0
    F = Chebfun2v.from_functions(lambda x, y: y + 0 * x,
                                 lambda x, y: -w ** 2 * x + 0 * y,
                                 domain=dom)
    rhs = _field_rhs(F)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for ic in np.arange(0.1, 1.01, 0.2):
        sol = solve_ivp(rhs, [0, 4], [ic, 0.0], rtol=1e-3, atol=1e-6,
                        dense_output=True)
        ts = np.linspace(0, 4, 400)
        Y = sol.sol(ts)
        ax.plot(Y[0], Y[1], "r", lw=1.0)
    _quiver(ax, F, dom)
    ax.set_title("The simple harmonic oscillator", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AutonomousSystems_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- Nonlinear pendulum ------------------------------------------
    dom = (-4.0, 4.0, -2.0, 2.0)
    F = Chebfun2v.from_functions(lambda x, y: y + 0 * x,
                                 lambda x, y: -jnp.sin(x) / 4 + 0 * y,
                                 domain=dom)
    rhs = _field_rhs(F)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for ic in np.arange(0.5, 3.01, 0.5):
        sol = solve_ivp(rhs, [0, 40], [ic, 0.0], rtol=1e-3, atol=1e-6,
                        dense_output=True)
        ts = np.linspace(0, 40, 1200)
        Y = sol.sol(ts)
        ax.plot(Y[0], Y[1], "r", lw=0.8)
    _quiver(ax, F, dom)
    ax.set_aspect("equal")
    ax.set_title("The eye of a nonlinear pendulum", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AutonomousSystems_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # -- Duffing oscillator ------------------------------------------
    d, a, b = 0.04, 1.0, -0.75
    dom = (-2.0, 2.0, -2.0, 2.0)
    F = Chebfun2v.from_functions(
        lambda x, y: y + 0 * x,
        lambda x, y: -d * y - b * x - a * x ** 3, domain=dom)
    rhs = _field_rhs(F)
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    sol = solve_ivp(rhs, [0, 40], [0.0, 0.5], rtol=1e-3, atol=1e-6,
                    dense_output=True)
    ts = np.linspace(0, 40, 2400)
    Y = sol.sol(ts)
    ax.plot(Y[0], Y[1], "r", lw=0.8)
    _quiver(ax, F, dom)
    ax.set_aspect("equal")
    ax.set_title("The Duffing oscillator", fontsize=14)

    r = np.asarray(F.roots())
    order = np.argsort(r[:, 0])
    r = r[order]
    print("r =")
    for row in r:
        print(f"  {row[0]: .15f}   {row[1]: 17.0f}"
              if row[1] == 0 else f"  {row[0]: .15f}   {row[1]: .15f}")
    ax.plot(r[:, 0], r[:, 1], "k.", ms=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "AutonomousSystems_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    run()
