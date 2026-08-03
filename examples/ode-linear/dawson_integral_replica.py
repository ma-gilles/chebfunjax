"""Dawson's integral.

Faithful replica of ode-linear/DawsonIntegral.m by Kuan Xu
(October 2012): the BVP

    dF/dx + 2xF = 1,   F(0) = 0

solved three ways: chebop with an interior point condition, the
analytic formula F = exp(-x^2) int_0^x exp(t^2) dt assembled with
cumsum/newDomain/join/merge, and Weideman's complex error function
routine.

Original: https://www.chebfun.org/examples/ode-linear/DawsonIntegral.html
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
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"DawsonIntegral_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cef(z, N):
    """Weideman's complex error function routine (numpy port)."""
    M = 2 * N
    M2 = 2 * M
    k = np.arange(-M + 1, M)
    L = np.sqrt(N / np.sqrt(2))
    theta = k * np.pi / M
    t = L * np.tan(theta / 2)
    f = np.exp(-t**2) * (L**2 + t**2)
    f = np.concatenate([[0.0], f])
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / M2
    a = a[1:N + 1][::-1]
    Z = (L + 1j * z) / (L - 1j * z)
    p = np.polyval(a, Z)
    return 2 * p / (L - 1j * z)**2 + (1 / np.sqrt(np.pi)) / (L - 1j * z)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    W, H = 5.0, 0.8

    # Chebop solve with interior point condition F(0) = 0
    t0 = time.time()
    L = Chebop(lambda x, f: f.diff(1) + 2 * x * f, domain=(-W, W))
    L.bc = lambda x, f: f(0.0)
    f = L.solve(1.0)
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    tt = np.linspace(-W, W, 1200)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(tt, np.asarray(f(tt)), lw=1.6)
    ax.axis([-W, W, -H, H])
    ax.grid(True)
    _save(fig)

    # Analytic construction: F = exp(-x^2) cumsum(exp(x^2)) on [0, W],
    # extended to [-W, 0] by odd symmetry.
    x = cj.chebfun(lambda t: t, domain=(0, W))
    fr = (-x**2).exp() * ((x**2).exp()).cumsum()
    fl = (-fr.flipud()).new_domain((-W, 0.0))
    f2 = fl.join(fr)
    f2 = f2.merge()
    print("f =")
    print(repr(f2))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(tt, np.asarray(f2(tt)), lw=1.6)
    ax.axis([-W, W, -H, H])
    ax.grid(True)
    _save(fig)

    # Weideman's rational approximation of the complex error function
    N = 36
    t0 = time.time()
    f3 = cj.chebfun(
        lambda t: jnp.asarray(np.real(
            np.sqrt(np.pi) * (_cef(np.asarray(t), N)
                              - np.exp(-np.asarray(t)**2)) / 2j)),
        domain=(-W, W))
    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(tt, np.asarray(f3(tt)), lw=1.6)
    ax.axis([-W, W, -H, H])
    ax.grid(True)
    _save(fig)


if __name__ == "__main__":
    run()
