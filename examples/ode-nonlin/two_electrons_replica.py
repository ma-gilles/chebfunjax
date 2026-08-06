"""Two electrons orbiting symmetrically about a nucleus.

Faithful replica of ode-nonlin/TwoElectrons.m: two electrons in mirror
orbits about a fixed nucleus, so one complex chebfun z(t) describes both
(the second is its conjugate). The equation is

    z'' + 2z/|z|^3 - (i/4) Im(z)/Im(z)^3 = 0,

the first term the attraction of the nucleus and the second the mutual
repulsion of the pair. Different initial speeds V give periodic orbits
of different periods.

Original: https://www.chebfun.org/examples/ode-nonlin/TwoElectrons.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style
from chebfunjax.tech.trigtech import Trigtech

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"TwoElectrons_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def solve(V, dom):
    N = Chebop(lambda t, z: z.diff(2) + 2 * z / abs(z) ** 3
               - 0.25j * z.imag() / z.imag() ** 3, domain=dom)
    N.lbc = [1j, V]
    return N.solve(0.0)


def orbit_plot(z, dom, lim=1.2):
    t = np.linspace(dom[0], dom[1], 8000)
    zv = np.asarray(z(t))
    x, y = zv.real, zv.imag
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    # MATLAB advances the axes ColorOrder even for the explicitly black
    # marker, so the mirror pair is drawn in colours 2 and 3, not 1 and
    # 2 -- the published figure is orange over yellow, not blue over
    # orange.
    ax.plot(0, 0, ".k")
    ax.plot(x, y, lw=0.7, color="#D95319")
    ax.plot(x, -y, lw=0.7, color="#EDB120")
    ax.axis(lim * np.array([-1, 1, -1, 1]))
    ax.set_aspect("equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    return fig, ax


def x_of_t(z, dom, ylab="x(t)", ylim=(-1.5, 1.5)):
    t = np.linspace(dom[0], dom[1], 8000)
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.plot(t, np.asarray(z(t)).real, lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel(ylab)
    if ylim:
        ax.set_ylim(*ylim)
    return fig, ax


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t_start = time.time()

    # --- V = 1 on [0, 40] --------------------------------------------
    dom = (0, 40)
    z = solve(1.0, dom)
    _save(orbit_plot(z, dom)[0])
    _save(x_of_t(z, dom)[0])

    # --- V = 1.446 on [0, 20]: period from successive local minima ---
    dom = (0, 20)
    z = solve(1.446, dom)
    _save(orbit_plot(z, dom)[0])
    _save(x_of_t(z, dom, ylab="x")[0])
    pos, _val = z.real().min(flag="local")
    pos = np.asarray(pos)
    T = float(pos[2] - pos[1])          # MATLAB pos(3) - pos(2)
    print(f"T =\n  {T:.15f}")

    # --- V = 0.783: period from where x crosses 0.9*max(x) upward ----
    z = solve(0.783, dom)
    _save(orbit_plot(z, dom)[0])
    x = z.real()
    xmax = float(x.max()[1])
    r = np.asarray((x - 0.9 * xmax).roots(), dtype=float)
    up = np.asarray(x.diff()(r))
    r = r[np.real(up) > 0]
    T = float(r[1] - r[0])
    print(f"T =\n  {T:.15f}")

    # --- V = 1.17745: a near-closed orbit, refined by one Newton step -
    V = 1.17745
    z = solve(V, dom)
    _save(orbit_plot(z, dom)[0])
    y = z.imag()
    r = np.asarray((y - 0.9999999).roots(), dtype=float)
    T = float(np.mean(r[-2:]))
    print(f"T =\n {T:.15f}")
    print(f"ans =\n  {complex(z(np.float64(T))):.15f}")
    T = T - float(np.real(complex(z(np.float64(T))))) / V
    print(f"T =\n {T:.15f}")
    print(f"ans =\n  {complex(z(np.float64(T))):.15f}")

    # One period as a trig chebfun: its coefficients and the velocities.
    zT = z.restrict(0.0, T)
    # Sample onto an equispaced grid first and build the trig
    # representation from values: constructing it adaptively straight
    # from the piecewise chebfun drives the jitted evaluator through
    # hundreds of distinct array shapes and XLA gives up with
    # "Failed to materialize symbols".
    npts = 2048
    tgrid = 0.0 + (T - 0.0) * np.arange(npts) / npts
    zvals = np.asarray(zT(tgrid))
    tech = Trigtech.from_values(jnp.asarray(zvals))
    zTtrig = Chebfun(funs=[_Piece(tech=tech, interval=(0.0, T))],
                     domain=Domain((0.0, T)))
    c = np.abs(np.asarray(zTtrig.trigcoeffs()))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.semilogy(np.arange(len(c)) - len(c) // 2, np.maximum(c, 1e-20),
                ".k", markersize=6)
    ax.set_xlabel("wave number")
    ax.set_ylabel("magnitude of Fourier coefficient")
    ax.grid(True)
    _save(fig)

    tt = np.linspace(0.0, T, 4000)
    dv = np.asarray(zTtrig.diff()(tt))
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.plot(dv.real, dv.imag, color="m", lw=0.8)
    ax.set_title("Velocities z'(t)")
    ax.axis([-3, 3, -3, 3])
    ax.set_aspect("equal")
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(-3, 4))
    _save(fig)

    # --- V = 0.13220442 on [0, 10]: a tiny orbit --------------------
    dom = (0, 10)
    z = solve(0.13220442, dom)
    _save(orbit_plot(z, dom)[0])
    fig, ax = orbit_plot(z, dom, lim=0.001)
    ax.set_xticks([-0.001, 0, 0.001])
    ax.set_yticks([-0.001, 0, 0.001])
    _save(fig)

    print(f"total_time_in_seconds =\n  {time.time() - t_start:.9f}")


if __name__ == "__main__":
    run()
