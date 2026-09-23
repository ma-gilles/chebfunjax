"""KdV solitons and non-solitons.

Translation of pde/KdV.m by Nick Trefethen (May 2016): the KdV
equation u_t = -0.5(u^2)_x - u_xxx solved with spin (ETDRK4, N = 800,
dt = 5e-6) on [0, 20] -- soliton overtaking, amplitude/speed laws,
non-soliton breakups, a soliton train, and four conserved quantities.

Original: https://www.chebfun.org/examples/pde/KdV.html
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
from chebfunjax.plotting import chebfun_style
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.spin.solver import spin
from chebfunjax.spin.spinop import SpinOp

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'pde')

A, B = 25.0, 23.0
DOM = (0.0, 20.0)
TMAX = 0.0156
N, DT = 800, 5e-6
FIG = [0]


def _kdv(init):
    """u = spin(S, N, dt): the final time as a trig chebfun."""
    op = SpinOp(lin_coeff=lambda k: -(1j * k) ** 3,
                nonlin_vals=lambda u: -0.5 * u**2,
                nonlin_diff_order=1,
                domain=DOM, tspan=(0.0, TMAX),
                u0=lambda x: np.asarray(init(jnp.asarray(x))))
    # MATLAB spin's default has NO dealiasing.
    _x, _t, u = spin(op, N, DT, dealias=False)
    return cj.chebfun(jnp.asarray(u), domain=DOM, trig=True)


def _plot(u0, u, labels=()):
    """plot(S.init), hold on, plot(u), hold off [, text(...)]"""
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    u0.plot(ax=ax, linewidth=1.5, n_pts=4000)
    u.plot(ax=ax, color="C1", linewidth=1.5, n_pts=4000)
    for xt, yt, txt in labels:
        ax.text(xt, yt, txt)
    ax.set_xlim(*DOM)
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, f"KdV_{FIG[0]:02d}.png"))
    plt.close(fig)


def _long(v):
    """A non-integer scalar in MATLAB format long (e-notation)."""
    return f"{v:26.15e}"


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    x = cj.chebfun(lambda t: t, domain=DOM)

    # Two solitons: the taller overtakes the slower.
    init = 3 * A**2 * (.5 * A * (x - 3)).sech()**2 \
        + 3 * B**2 * (.5 * B * (x - 4)).sech()**2
    t0 = time.time()
    u = _kdv(init)
    time_in_seconds = time.time() - t0
    _plot(init, u, ((4.4, 1300, 't = 0'), (13.5, 1300, 't = 0.0156')))
    print("time_in_seconds =")
    print(f"   {time_in_seconds:.15f}")

    # Amplitude and speed of a single soliton.
    init = 3 * A**2 * (.5 * A * (x - 3)).sech()**2
    u = _kdv(init)
    _plot(init, u, ((3.4, 1300, 't = 0'), (13.2, 1300, 't = 0.0156')))
    print("initial_amplitude =")
    print(f"        {3 * A**2:.0f}")
    pos, val = u.max()
    print("final_amplitude =")
    print(_long(float(val)))
    print("predicted_speed =")
    print(f"   {A**2:.0f}")
    print("observed_speed =")
    print(_long((float(pos) - 3) / TMAX))

    # Non-soliton solutions.
    init = 3 * A**2 * (.35 * A * (x - 3)).sech()**2
    u = _kdv(init)
    _plot(init, u)

    init = 3 * A**2 * ((.05 * A * (x - 3)).sech()**2
                       + (.05 * A * (x - 23)).sech()**2)
    u = _kdv(init)
    _plot(init, u)

    init = 500 * (x - 12) * (-(x - 12)**2).exp()
    u = _kdv(init)
    _plot(init, u)

    # Conserved quantities.
    u0 = init
    for name, text, fn in (
        ("conserved1", "@(u)sum(u)", lambda v: v.sum()),
        ("conserved2", "@(u)sum(u.^2)", lambda v: (v**2).sum()),
        ("conserved3", "@(u)sum(u.^3/3-diff(u).^2)",
         lambda v: (v**3 / 3 - v.diff()**2).sum()),
        ("conserved4", "@(u)sum(u.^4/4-3*u.*diff(u).^2+(9/5)*diff(u,2).^2)",
         lambda v: (v**4 / 4 - 3 * v * v.diff()**2
                    + (9 / 5) * v.diff(2)**2).sum()),
    ):
        print(f"{name} = ")
        print(f"    {text}")
        print("ans =")
        print(_long(float(fn(u))))
        print("ans =")
        print(_long(float(fn(u0))))


if __name__ == "__main__":
    run()
