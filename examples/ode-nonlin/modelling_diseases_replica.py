"""Modelling diseases.

Faithful replica of ode-nonlin/ModellingDiseases.m by Toby Driscoll
(November 2014): the SIR epidemic model solved as a nonlinear IVP
system, its peak infection count, the crossover time where the
infected and recovered populations are equal, and the instantaneous
mortality rate.

Original: https://www.chebfun.org/examples/ode-nonlin/ModellingDiseases.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
CONTACT, RECOVERY = 0.003, 0.3


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ModellingDiseases_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    N = Chebop(lambda x, S, I, R: [
        S.diff() + CONTACT * I * S,
        I.diff() - CONTACT * I * S + RECOVERY * I,
        R.diff() - RECOVERY * I], domain=(0, 30))
    N.lbc = lambda S, I, R: [S - 500, I - 1, R]
    S, I, R = N.solve(0.0)

    t = np.linspace(0, 30, 2000)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for c, lab in ((S, "S"), (I, "I"), (R, "R")):
        ax.plot(t, np.asarray(c(t)), lw=1.6, label=lab)
    ax.legend()
    ax.set_title("SIR model")
    ax.set_xlabel("t")
    ax.grid(True)
    _save(fig)

    mx = I.max()
    peak = float(mx[1]) if isinstance(mx, tuple) else float(mx)
    print("ans =")
    print(f"   {round(peak)}")

    t_eq = np.asarray((I - R).roots(), dtype=float).ravel()
    t_eq = float(t_eq[0])
    print("t_eq =")
    print(f"   {t_eq:.15f}")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for c, lab in ((S, "S"), (I, "I"), (R, "R")):
        ax.plot(t, np.asarray(c(t)), lw=1.6, label=lab)
    ax.legend()
    ax.set_xlabel("t")
    ylim = ax.get_ylim()
    ax.plot([t_eq, t_eq], ylim, 'k:')
    ax.plot(t_eq, float(I(jnp.asarray(t_eq))), 'k.', ms=13)
    ax.set_ylim(*ylim)
    ax.grid(True)
    _save(fig)

    rho = 0.4                      # 40 percent of infected people die
    mortality = rho * R / I.cumsum()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    tm = np.linspace(1e-6, 30, 2000)
    ax.plot(tm, np.asarray(mortality(tm)), lw=1.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("t")
    ax.set_title("Instantaneous mortality rate for the SIR model")
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
