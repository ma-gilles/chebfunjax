"""KdV solitons and non-solitons.

Faithful replica of pde/KdV.m by Nick Trefethen (May 2016): the KdV
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

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.plotting import chebfun_style
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


def _kdv(u0):
    op = SpinOp(lin_coeff=lambda k: -(1j * k) ** 3,
                nonlin_vals=lambda u: -0.5 * u**2,
                nonlin_diff_order=1,
                domain=DOM, tspan=(0.0, TMAX), u0=u0)
    # MATLAB spin's default has NO dealiasing; with our dealias=True
    # default the sharp soliton's peak lands 10 digits away from the
    # published amplitude instead of matching it.
    return spin(op, N, DT, dealias=False)


def _trig_interp(u, xf):
    """Evaluate the trig interpolant of grid values u at points xf."""
    c = np.fft.fft(u) / len(u)
    k = np.fft.fftfreq(len(u), d=1.0 / len(u))
    L = DOM[1] - DOM[0]
    ph = np.exp(2j * np.pi * np.outer(xf - DOM[0], k) / L)
    return (ph @ c).real


def _plot(x, u0v, uv):
    FIG[0] += 1
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.plot(x, u0v, lw=1.4)
    ax.plot(x, uv, lw=1.4)
    ax.set_xlim(*DOM)
    ax.grid(True)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, f"KdV_repl_{FIG[0]:02d}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    # 1. Two solitons: the taller overtakes the slower.
    def u0_two(x):
        return (3 * A**2 / np.cosh(.5 * A * (x - 3))**2
                + 3 * B**2 / np.cosh(.5 * B * (x - 4))**2)

    t0 = time.time()
    x, t, u = _kdv(u0_two)
    print("time_in_seconds =")
    print(f"   {time.time() - t0:.9f}")
    _plot(x, u0_two(x), u)

    # 2. Amplitude and speed of a single soliton.
    def u0_one(x):
        return 3 * A**2 / np.cosh(.5 * A * (x - 3))**2

    x, t, u = _kdv(u0_one)
    _plot(x, u0_one(x), u)
    print("initial_amplitude =")
    print(f"        {3 * A**2:.0f}")
    xf = np.linspace(*DOM, 200001)
    uf = _trig_interp(u, xf)
    pos0 = xf[np.argmax(uf)]
    xz = np.linspace(pos0 - 2e-4, pos0 + 2e-4, 40001)
    uz = _trig_interp(u, xz)
    val, pos = np.max(uz), xz[np.argmax(uz)]
    print("final_amplitude =")
    print(f"     {val:.15e}")
    print("predicted_speed =")
    print(f"   {A**2:.0f}")
    print("observed_speed =")
    print(f"     {(pos - 3) / TMAX:.15e}")

    # 3. Non-soliton solutions.
    def u0_wide(x):
        return 3 * A**2 / np.cosh(.35 * A * (x - 3))**2

    x, t, u = _kdv(u0_wide)
    _plot(x, u0_wide(x), u)

    def u0_train(x):
        return 3 * A**2 * (1 / np.cosh(.05 * A * (x - 3))**2
                           + 1 / np.cosh(.05 * A * (x - 23))**2)

    x, t, u = _kdv(u0_train)
    _plot(x, u0_train(x), u)

    def u0_rand(x):
        return 500 * (x - 12) * np.exp(-(x - 12)**2)

    x, t, u = _kdv(u0_rand)
    _plot(x, u0_rand(x), u)

    # 4. Conservation laws (trapezoid = exact for periodic grids).
    L = DOM[1] - DOM[0]
    h = L / N
    k = np.fft.fftfreq(N, d=1.0 / N) * 2 * np.pi / L

    def deriv(v, order=1):
        return np.real(np.fft.ifft((1j * k) ** order * np.fft.fft(v)))

    u0v = u0_rand(x)
    for name, fn in [
        ("conserved1", lambda v: h * np.sum(v)),
        ("conserved2", lambda v: h * np.sum(v**2)),
        ("conserved3", lambda v: h * np.sum(v**3 / 3 - deriv(v)**2)),
        ("conserved4", lambda v: h * np.sum(
            v**4 / 4 - 3 * v * deriv(v)**2 + 9 / 5 * deriv(v, 2)**2)),
    ]:
        print(f"{name}: u = {fn(u):.12e}   u0 = {fn(u0v):.12e}")


if __name__ == "__main__":
    run()
