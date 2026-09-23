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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, phaseplot, surf
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.spin.solver2d import spin2 as _spin2
from chebfunjax.spin.spinop2 import SpinOp2

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')
FIG = [0]

# S = spinop2('gl'): MATLAB's display of the built-in operator.  Our
# SpinOp2 has no MATLAB-style display; the text is Chebfun's.
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
    """spinop2(dom, tspan) with S.lin = @(u) lap(u) and
    S.nonlin = @(u) u - (1+1.5i)*u.*(abs(u).^2)."""
    return SpinOp2(lin_coeffs=(1.0, 0.0, 0.0, 0.0, 0.0),
                   nonlin_vals=lambda u: u - (1 + 1.5j) * u * jnp.abs(u) ** 2,
                   n_vars=1, domain=dom, tspan=tspan, u0=None, is_real=False)


def _values2chebfun2(U, dom):
    """chebfun2(vals, dom, 'trig') for the complex solution values.

    cj.chebfun2 of a COMPLEX value matrix with trig=True returns wrong
    values, so the real and imaginary parts are built separately."""
    V = np.asarray(U).T                      # rows = y, as MATLAB
    return (cj.chebfun2(V.real, domain=dom, trig=True)
            + 1j * cj.chebfun2(V.imag, domain=dom, trig=True))


def spin2(S, npts, dt):
    """u = spin2(S, npts, dt, 'plot', 'off').  For tspan = [t0 t1 ... tk]
    returns the list of chebfun2s at t0, t1, ..., tk (MATLAB's u{j})."""
    ts = [float(v) for v in S.tspan]
    out = [S.u0]
    for a, b in zip(ts[:-1], ts[1:]):
        op = SpinOp2(S.lin_coeffs, S.nonlin_vals, 1, S.domain, (a, b),
                     out[-1], is_real=False)
        _, _, _, U = _spin2(op, npts, dt, dealias=False)
        out.append(_values2chebfun2(U, S.domain))
    return out[-1] if len(ts) == 2 else out


def _plot(u):
    """plot(u), view(0,90), axis equal, axis off."""
    FIG[0] += 1
    fig, ax = surf(u, n_pts=200)
    ax.view_init(elev=90, azim=-90)
    ax.set_box_aspect((1, 1, 1e-3))
    ax.set_axis_off()
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, f"GinzburgLandau_{FIG[0]:02d}.png"))
    plt.close(fig)


def _plot_phase(u):
    """plot(u) for a complex chebfun2: phase portrait of angle(-u)."""
    FIG[0] += 1
    dom = [float(v) for v in u.domain]
    fig, ax = phaseplot(lambda z: -np.asarray(u(jnp.asarray(z.real),
                                                 jnp.asarray(z.imag))),
                        region=dom, n_pts=500)
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

    S = SpinOp2.from_name('gl')
    print("S = ")
    print(S_GL_DISPLAY, flush=True)

    dom = (-50.0, 50.0, -50.0, 50.0)
    tspan = (0.0, 16.0)
    S = spinop2(dom, tspan)

    x = cj.chebfun2(lambda x, y: x, domain=dom)
    y = cj.chebfun2(lambda x, y: y, domain=dom)
    u1 = (1j * x + y) * cj.exp(-.03 * (x ** 2 + y ** 2))
    S.u0 = u1
    npts = 80
    dt = 4 / npts
    t0 = time.perf_counter()
    u = spin2(S, npts, dt)
    _plot(u.real())

    u2 = (x + y) * cj.exp(-.03 * (x ** 2 + y ** 2))
    S.u0 = u2
    u = spin2(S, npts, dt)
    _plot(u.real())
    _toc(t0)

    tspan = (0.0, 48.0)
    S = spinop2(dom, tspan)
    S.u0 = u1
    t0 = time.perf_counter()
    u = spin2(S, npts, dt)
    _plot(u.real())

    S.u0 = u2
    u = spin2(S, npts, dt)
    _plot(u.real())
    _toc(t0)

    tspan = (0.0, 96.0)
    S = spinop2(dom, tspan)
    S.u0 = u1
    u = spin2(S, npts, dt)
    _plot(u.real())

    npts = 128
    dt = 4 / npts
    t0 = time.perf_counter()
    S.u0 = u2
    u = spin2(S, npts, dt)
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
    S.u0 = u1
    npts = 128
    dt = 8 / npts
    t0 = time.perf_counter()
    u = spin2(S, npts, dt)
    _plot(u[1].real())
    _plot(u[2].real())
    _plot_phase(u[2])
    _toc(t0)


if __name__ == "__main__":
    run()
