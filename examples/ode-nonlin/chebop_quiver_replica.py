"""Phase portraits with chebop/quiver.

Faithful replica of ode-nonlin/ChebopQuiver.m by Asgeir Birkisson
(November 2015): direction fields drawn by ``Chebop.quiver`` for the
van der Pol oscillator, the undamped and damped nonlinear pendulum, and
the Lotka-Volterra predator-prey equations, with particular solutions
overlaid.

Original: https://www.chebfun.org/examples/ode-nonlin/ChebopQuiver.html
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

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import arrowplot, chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]

# MATLAB's default axes ColorOrder. The quiver plot is drawn first and
# takes index 0 (blue), so overlaid solutions start at index 1.
MATLAB_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E",
                 "#77AC30", "#4DBEEE", "#A2142F"]


def _curve_color(k):
    return MATLAB_COLORS[(k + 1) % len(MATLAB_COLORS)]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"ChebopQuiver_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # --- The van der Pol equation -----------------------------------
    N = Chebop(lambda t, u: u.diff(2) - 3 * (1 - u**2) * u.diff() + u,
               domain=(0, 100))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    N.quiver([-2.75, 2.75, -5.5, 5.5], ax=ax, xpts=40, ypts=40,
             scale=.5, normalize=True)
    for k, init in enumerate((0.2,)):          # MATLAB: 0.2:0.4:0.2
        N.lbc = [init, 1]
        u = N.solve(0.0)
        arrowplot(u, u.diff(), ax=ax, color=_curve_color(k), linewidth=1.6)
    ax.set_title("Phase portrait of the van der Pol oscillator")
    ax.set_xlabel("$u$")
    ax.set_ylabel("$u'$")
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-5.5, 5.5)
    _save(fig)

    # --- An undamped mathematical pendulum ---------------------------
    N = Chebop(lambda t, u: u.diff(2) + u.sin(), domain=(0, 50))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    N.quiver([-2.5, 25, -2, 5.5], ax=ax, xpts=30)
    for k, init in enumerate(np.arange(0, 5.01, 0.5)):
        N.lbc = [0, float(init)]
        u = N.solve(0.0)
        arrowplot(u, u.diff(), ax=ax, color=_curve_color(k), linewidth=1.4)
    ax.set_xlim(-2.5, 25)
    ax.set_ylim(-2, 5.5)
    ax.set_title("Phase portrait for an undamped nonlinear pendulum")
    ax.set_xlabel("$u$")
    ax.set_ylabel("$u'$")
    _save(fig)

    # --- A damped pendulum: every trajectory comes to rest -----------
    N = Chebop(lambda t, u: u.diff(2) + 0.25 * u.diff() + u.sin(),
               domain=(0, 50))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    N.quiver([-2.5, 25, -2, 5.5], ax=ax, xpts=30)
    for k, init in enumerate(np.arange(0, 5.01, 0.5)):
        N.lbc = [0, float(init)]
        u = N.solve(0.0)
        t = np.linspace(0, 50, 3000)
        ax.plot(np.asarray(u(t)), np.asarray(u.diff()(t)),
                color=_curve_color(k), lw=1.4)
    ax.set_xlim(-2.5, 25)
    ax.set_ylim(-2, 5.5)
    ax.set_title("Phase portrait for a damped nonlinear pendulum")
    ax.set_xlabel("$u$")
    ax.set_ylabel("$u'$")
    _save(fig)

    # --- Lotka-Volterra predator-prey --------------------------------
    for growth, title in (
            (1.0, "Phase portrait for Lotka-Volterra equations"),
            (1.5, "Phase portrait for L-V eqns., "
                  "increased rabbit reproduction")):
        N = Chebop(lambda t, u, v, _a=growth: [u.diff() - _a * u + u * v,
                                               v.diff() + v - u * v],
                   domain=(0, 10))
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        N.quiver([0, 5, 0, 5], ax=ax, xpts=30, ypts=30,
                 normalize=True, scale=.4)
        for k, rabbits in enumerate(np.arange(0.1, 1.91, 0.2)):
            N.lbc = (lambda u, v, _r=float(rabbits):
                     [u - _r, v - 1])
            u, v = N.solve(0.0)
            arrowplot(u, v, ax=ax, color=_curve_color(k), linewidth=1.4)
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_title(title)
        ax.set_xlabel("Rabbits")
        ax.set_ylabel("Foxes")
        _save(fig)


if __name__ == "__main__":
    run()
