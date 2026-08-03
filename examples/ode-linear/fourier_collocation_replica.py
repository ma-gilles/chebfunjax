"""Fourier collocation for periodic ODEs.

Faithful replica of ode-linear/FourierCollocation.m by Hadrien
Montanelli (December 2014): periodic linear ODEs solved with the
Fourier (trigonometric) discretization, compared against Chebyshev
collocation with wrap-around rows, plus the Hill (Mathieu-like)
discriminant computed from two initial-value solves.

Original: https://www.chebfun.org/examples/ode-linear/FourierCollocation.html
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

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-linear')

FIG = [0]
DOM = (0, 2 * np.pi)


def _save(fig):
    FIG[0] += 1
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(
        _IMG, f"FourierCollocation_repl_{FIG[0]:02d}.png"),
        dpi=150, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t = np.linspace(DOM[0], DOM[1], 1600)
    xf = cj.chebfun(lambda s: s, domain=DOM)

    # First-order periodic problem
    L = Chebop(lambda x, u: u.diff() + (1 + (10 * x).cos().sin()) * u,
               domain=DOM)
    L.bc = "periodic"
    f = cj.chebfun(lambda s: jnp.exp(jnp.sin(s)), domain=DOM)
    u = L.solve(f)
    print("u =")
    print(repr(u))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u(t)), lw=2)
    ax.grid(True)
    ru = (u.diff() + (1 + (10 * xf).cos().sin()) * u - f).norm(jnp.inf)
    print("ans =")
    print(f"     {float(ru):.15e}")

    v = L.solve(f, discretization="chebcolloc2")
    print("v =")
    print(repr(v))
    ax.plot(t, np.asarray(v(t)), 'r', lw=2)
    _save(fig)
    rv = (v.diff() + (1 + (10 * xf).cos().sin()) * v - f).norm(jnp.inf)
    print("ans =")
    print(f"     {float(rv):.15e}")
    print("ans =")
    print(f"   {len(v) / len(u):.15f}")

    # Second-order periodic problem
    a1 = cj.chebfun(lambda s: jnp.sin(jnp.cos(s / 2)**2), domain=DOM)
    a0 = cj.chebfun(lambda s: jnp.cos(12 * jnp.sin(s)), domain=DOM)
    L2 = Chebop(lambda x, u: u.diff(2) + a1 * u.diff() + a0 * u,
                domain=DOM)
    L2.bc = "periodic"
    f2 = cj.chebfun(lambda s: jnp.exp(jnp.cos(2 * s)), domain=DOM)
    u2 = L2.solve(f2)
    print("u =")
    print(repr(u2))
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(t, np.asarray(u2(t)), lw=2)
    ax.grid(True)
    ru2 = (u2.diff(2) + a1 * u2.diff() + a0 * u2 - f2).norm(jnp.inf)
    print("ans =")
    print(f"     {float(ru2):.15e}")

    v2 = L2.solve(f2, discretization="chebcolloc2")
    print("v =")
    print(repr(v2))
    ax.plot(t, np.asarray(v2(t)), 'r', lw=2)
    _save(fig)
    print("ans =")
    print(f"   {len(v2) / len(u2):.15f}")

    # Hill discriminant via two IVP solves
    def _mk(lbc):
        Lh = Chebop(lambda x, u: u.diff(2)
                    + ((x / 2).cos()**2).sin() * u.diff()
                    + (12 * x.sin()).cos() * u, domain=DOM)
        Lh.lbc = lbc
        return Lh

    c = _mk(lambda c: [c - 1, c.diff()]).solve(0.0)
    s = _mk(lambda s: [s, s.diff() - 1]).solve(0.0)
    HillDiscr = 0.5 * (float(c(jnp.array(2 * np.pi)))
                       + float(s.diff()(jnp.array(2 * np.pi))))
    print("HillDiscr =")
    print(f"   {HillDiscr:.15f}")


if __name__ == "__main__":
    run()
