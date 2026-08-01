"""Periodic CF approximation: Eureka!

Faithful replica of fourier/TrigCFExample.m by Nick Trefethen,
February 2017: the singular values of a Hankel matrix of Fourier
coefficients predict the minimax error of periodic rational
approximation (a periodic Caratheodory-Fejer bound), matching
trigremez to many digits.

Original: https://www.chebfun.org/examples/fourier/TrigCFExample.html
Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
from scipy.special import factorial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fourier')


def _pc(vals):
    for v in np.atleast_1d(vals):
        v = complex(v)
        sign = "+" if v.imag >= 0 else "-"
        print(f"  {v.real: .15f} {sign} {abs(v.imag):.15f}i")


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()

    # A warm-up Hankel norm (the page's opening one-liner).
    print("ans =")
    print(f"   {np.linalg.norm(sla.hankel(1.0 / factorial(np.arange(3, 10))), 2):.15f}")

    f = cj.chebfun(lambda t: jnp.exp(jnp.sin(t)),
                   domain=[-np.pi, np.pi], trig=True)
    c = np.asarray(f.trigcoeffs())
    print("c =")
    _pc(c)
    c = c[(len(c) + 1) // 2 - 1:]
    print("c =")
    _pc(c)

    m, n = 2, 1
    H = sla.hankel(c[1 + m - n:])
    print("ans =")
    for row in np.round(np.real(H[:3, :3]) + 1j * np.imag(H[:3, :3]), 4):
        print("  " + "   ".join(
            f"{v.real: .4f} {'+' if v.imag >= 0 else '-'} "
            f"{abs(v.imag):.4f}i" for v in row))
    s = np.linalg.svd(H, compute_uv=False)
    print("ans =")
    print(f"   {2 * s[n]:.15f}")

    p, q, r, err, status = cj.trigremez(f, m, n)
    print("err =")
    print(f"   {err:.15f}")

    xs = np.linspace(-np.pi, np.pi, 3000)
    e = np.asarray(f(jnp.asarray(xs))) - np.asarray(r(xs))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, e, lw=1.6)
    ax.grid(True)
    ax.plot([-np.pi, np.pi], [err, err], "--k", lw=1.0)
    ax.plot([-np.pi, np.pi], [-err, -err], "--k", lw=1.0)
    ax.set_ylim(-0.004, 0.004)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "TrigCFExample_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")
    return True


if __name__ == "__main__":
    run()
