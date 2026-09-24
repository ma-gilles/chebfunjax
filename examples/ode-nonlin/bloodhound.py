"""Bloodhound supersonic car.

Translation of ode-nonlin/Bloodhound.m: the acceleration phase of
the Bloodhound SSC land-speed-record car. A jet burns from t = 0; the
rocket ignites at t = 11 s, so the mass and thrust coefficients are
piecewise, and the momentum balance

    m(t) v' + m'(t) v = T(t) - c_d v^2 - mu m(t) g

is a first-order nonlinear ODE with a kink at the ignition.

Original: https://www.chebfun.org/examples/ode-nonlin/Bloodhound.html
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

from chebfunjax import chebfun
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
DOM = (0.0, 50.0)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"Bloodhound_{FIG[0]:02d}.png"))
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    RocketStart = 11.0          # seconds
    MassOriginal = 6500.0       # kg
    JetThrust = 80.0            # kN
    RocketThrust = 115.0        # kN
    JSFC = 0.0005 * 102         # kg/(kN h)
    RSFC = 102 / 220

    def mass(t):
        return (MassOriginal - JSFC * JetThrust * t
                - RSFC * RocketThrust * (t - RocketStart)
                * (t > RocketStart))

    def thrust(t):
        return 1000 * (JetThrust + RocketThrust * (t > RocketStart))

    cmass = chebfun(mass, domain=DOM, splitting=True)
    cthrust = chebfun(thrust, domain=DOM, splitting=True)
    dcmass = cmass.diff()
    surfacedrag = (2 / 5) * cmass * 9.81

    N = Chebop(lambda t, v: cmass * v.diff() + dcmass * v - cthrust
               + (175 / 289) * v**2 + surfacedrag, domain=DOM)
    # MATLAB writes N.bc = @(t,v) v(0); the equivalent left boundary
    # condition routes to the piecewise solver, which places the
    # breakpoint at the rocket ignition.
    N.lbc = 0.0
    N.init = chebfun(lambda t: t, domain=DOM)
    u = N.solve(0.0)

    t1000 = float(np.asarray((u - 447.0).roots(), dtype=float)[0])

    tt = np.linspace(*DOM, 4000)
    u_mph = np.asarray(u(tt)) / 0.44704
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(tt, u_mph, lw=2)
    ax.plot([0, 50], [1000, 1000], "r", lw=2)
    ax.text(20, 500, f"Time to 1000 mph = {t1000:.3g} seconds")
    ax.set_title("Velocity of the Bloodhound Supersonic Car "
                 "(Acceleration Phase)")
    ax.set_ylabel("velocity in mph")
    ax.grid(True)
    _save(fig)

    s = u.cumsum()
    s_miles = np.asarray(s(tt)) / 1609
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.plot(tt, s_miles, lw=2)
    ax.set_title("Distance travelled")
    ax.set_xlabel("Time in seconds")
    ax.set_ylabel("Distance in miles")
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
