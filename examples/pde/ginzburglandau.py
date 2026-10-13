"""Complex Ginzburg-Landau equation in 2D.

Translation of pde/GinzburgLandau.m by Nick Trefethen (May
2016): the complex Ginzburg-Landau equation

    u_t = Lap u + u - (1 + 1.5i) u |u|^2

with spin2/ETDRK4 on [-50,50]^2: spirals from complex and real
initial data at t = 16, the beginnings of chaos at t = 48 (with the
diagonal symmetry preserved), chaos at t = 96, and the big-canvas
[-100,100]^2 two-spiral run to t = 30/60 plus the phase portrait
(complex-argument coloring).

Original: https://www.chebfun.org/examples/pde/GinzburgLandau.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax._colormaps import parula_colormap
from chebfunjax.operators.spinop2 import Spinop2, spin2
from chebfunjax.plotting import chebfun_style, matlab_axis_equal, matlab_view, surf
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')
FIG = [0]

# S = spinop2('gl'): MATLAB's display of the built-in operator. The
# text below is Chebfun's display for the public Spinop2 object.
S_GL_DISPLAY = """\
  spinop2 with properties:

     domain: [0 100 0 100]
       init: [InfxInf chebfun2]
        lin: @(u)lap(u)
     nonlin: @(u)u-(1+1.5i)*u.*(abs(u).^2)
      tspan: [0 100]
    numVars: 1
"""


def spinop2(dom, tspan):
    """Create the public Spinop2 for the Ginzburg-Landau equation."""
    S = Spinop2(dom, tspan)
    S.lin = "@(u)lap(u)"
    S.nonlin = "@(u)u-(1+1.5i)*u.*(abs(u).^2)"
    return S


def _plot(u):
    """plot(u), view(0,90), axis equal, axis off."""
    FIG[0] += 1
    # Official R2016a table at the historical64-color length: matched-field
    # controls support reference compatibility, not the cached PNG's release.
    fig, ax = surf(u, n_pts=200, cmap=parula_colormap(64, version="R2016a"))
    matlab_view(ax, 0, 90)
    matlab_axis_equal(ax)
    ax.set_axis_off()
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, f"GinzburgLandau_{FIG[0]:02d}.png"))
    plt.close(fig)


def _plot_phase(u):
    """plot(u) for a complex chebfun2: phase portrait of angle(-u)."""
    FIG[0] += 1
    fig, ax = u.plot()
    ax.set_axis_off()
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, f"GinzburgLandau_{FIG[0]:02d}.png"))
    plt.close(fig)


def _toc(t0):
    print("time_in_seconds =")
    print(f"{time.perf_counter() - t0:20.15f}", flush=True)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    S = Spinop2('gl')
    print("S = ")
    print(S_GL_DISPLAY, flush=True)

    dom = (-50.0, 50.0, -50.0, 50.0)
    tspan = (0.0, 16.0)
    S = spinop2(dom, tspan)

    x = cj.chebfun2(lambda x, y: x, domain=dom)
    y = cj.chebfun2(lambda x, y: y, domain=dom)
    u1 = (1j * x + y) * cj.exp(-.03 * (x ** 2 + y ** 2))
    S.init = u1
    npts = 80
    dt = 4 / npts
    t0 = time.perf_counter()
    u = spin2(S, npts, dt, 'plot', 'off')
    _plot(u.real())

    u2 = (x + y) * cj.exp(-.03 * (x ** 2 + y ** 2))
    S.init = u2
    u = spin2(S, npts, dt, 'plot', 'off')
    _plot(u.real())
    _toc(t0)

    tspan = (0.0, 48.0)
    S = spinop2(dom, tspan)
    S.init = u1
    t0 = time.perf_counter()
    u = spin2(S, npts, dt, 'plot', 'off')
    _plot(u.real())

    S.init = u2
    u = spin2(S, npts, dt, 'plot', 'off')
    _plot(u.real())
    _toc(t0)

    tspan = (0.0, 96.0)
    S = spinop2(dom, tspan)
    S.init = u1
    u = spin2(S, npts, dt, 'plot', 'off')
    _plot(u.real())

    npts = 128
    dt = 4 / npts
    t0 = time.perf_counter()
    S.init = u2
    u = spin2(S, npts, dt, 'plot', 'off')
    _plot(u.real())
    _toc(t0)

    dom = (-100.0, 100.0, -100.0, 100.0)
    tspan = (0.0, 30.0, 60.0)
    S = spinop2(dom, tspan)
    x = cj.chebfun2(lambda x, y: x, domain=dom)
    y = cj.chebfun2(lambda x, y: y, domain=dom)
    u1 = ((1j * (x - 8) + (y - 2))
          * cj.exp(-.03 * ((x - 8) ** 2 + (y - 2) ** 2))
          + ((x + 8) - (y + 2)) * cj.exp(-.03 * ((x + 8) ** 2 + (y + 2) ** 2)))
    S.init = u1
    npts = 128
    dt = 8 / npts
    t0 = time.perf_counter()
    u = spin2(S, npts, dt, 'plot', 'off')
    _plot(u[0, 1].real())
    _plot(u[0, 2].real())
    _plot_phase(u[0, 2])
    _toc(t0)


if __name__ == "__main__":
    run()
