"""Phase and argument of complex functions.

Faithful replica of complex/Arguments.m by Nick Trefethen (May 2011):
angle vs unwrapped argument for a spiral f(t) = t exp(it), and the
two ways of taking its square root.

Original: https://www.chebfun.org/examples/complex/Arguments.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'complex')


def _disp(v):
    print("ans =")
    print(f"  {v:.15f}" if v < 0 else (f"     {v:.0f}" if v == 0
                                       else f"   {v:.15f}"))


def run():
    os.makedirs(_IMG, exist_ok=True)

    _disp(float(np.angle(1)))
    _disp(float(np.angle(-1)))
    _disp(float(np.angle(-1 - 0.01j)))

    f = cj.chebfun(lambda t: t * jnp.exp(1j * t), domain=(1.0, 20.0))
    ts = np.linspace(1, 20, 6000)
    fv = np.asarray(f(jnp.asarray(ts)))

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    ax.plot(fv.real, fv.imag, lw=1.6)
    ax.set_aspect("equal")
    ax.set_title("f(t) in complex plane", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Arguments_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    ang = np.angle(fv)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    # draw each continuous branch separately (the jumps at +-pi)
    jumps = np.nonzero(np.abs(np.diff(ang)) > np.pi)[0]
    start = 0
    for j in list(jumps) + [len(ts) - 1]:
        ax.plot(ts[start:j + 1], ang[start:j + 1], 'm', lw=1.6)
        start = j + 1
    ax.set_xlabel("t")
    ax.set_ylabel("angle(f(t))")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Arguments_repl_02.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("ans =")
    u = np.unwrap(np.angle(np.array([-1, -1 - 0.01j])))
    print("   " + "   ".join(f"{v:.15f}" for v in u))

    arg = np.unwrap(ang)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(ts, arg, 'm', lw=1.6)
    ax.set_ylim(-1, 21)
    ax.set_xlabel("t")
    ax.set_ylabel("argument")
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Arguments_repl_03.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # sqrt via principal branch (with jumps)...
    gv = np.sqrt(np.abs(fv)) * np.exp(0.5j * ang)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    jumps = np.nonzero(np.abs(np.diff(np.angle(gv))) > 2)[0]
    start = 0
    for j in list(jumps) + [len(ts) - 1]:
        ax.plot(gv.real[start:j + 1], gv.imag[start:j + 1], 'C0',
                lw=1.6)
        start = j + 1
    ax.axis([-5, 5, -5, 5])
    ax.set_aspect("equal")
    ax.set_title("sqrt(f(t)) in complex plane", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Arguments_repl_04.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ... and via the unwrapped argument: a smooth spiral
    g2 = np.sqrt(np.abs(fv)) * np.exp(0.5j * arg)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.plot(g2.real, g2.imag, lw=1.6)
    ax.axis([-5, 5, -5, 5])
    ax.set_aspect("equal")
    ax.set_title("sqrt(f(t)) in complex plane", fontsize=14)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "Arguments_repl_05.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
