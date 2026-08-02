"""A quiz about nonnormal matrices.

Faithful replica of linalg/NonnormalQuiz.m by Nick Trefethen
(March 2015): transient growth of ||exp(tA)|| for two stable 2x2
matrices — the one that looks more dangerous is the tamer one.

Original: https://www.chebfun.org/examples/linalg/NonnormalQuiz.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')


def _expnorm_chebfun(A):
    def op(t_arr):
        t_arr = np.atleast_1d(np.asarray(t_arr, dtype=float))
        out = np.empty_like(t_arr)
        for i, t in enumerate(t_arr.ravel()):
            out.ravel()[i] = np.linalg.norm(sla.expm(t * A), 2)
        return out.reshape(np.shape(t_arr))
    return cj.chebfun(lambda t: jnp.asarray(op(np.asarray(t))),
                      domain=(0.0, 3.4))


def run():
    os.makedirs(_IMG, exist_ok=True)

    A1 = np.array([[-1, 1], [0, -1]], dtype=float)
    A2 = np.array([[-1, 5], [0, -2]], dtype=float)
    for name, A in (("A1", A1), ("A2", A2)):
        print(f"{name} =")
        for row in A:
            print("  " + "".join(f"{int(v):6d}" for v in row))

    e1 = _expnorm_chebfun(A1)
    e2 = _expnorm_chebfun(A2)
    xs = np.linspace(0, 3.4, 700)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(xs, np.asarray(e1(xs)), lw=1.6, label="A1")
    ax.plot(xs, np.asarray(e2(xs)), lw=1.6, label="A2")
    ax.set_ylim(0, 1.5)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|e^{tA}\|$")
    ax.set_title("Which curve is which?", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_IMG, "NonnormalQuiz_repl_01.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    p1, m1 = e1.max()
    print("maxnorm1 =")
    print(f"     {float(m1):g}")
    print("maxt1 =")
    print(f"     {float(p1):g}")
    p2, m2 = e2.max()
    print("maxnorm2 =")
    print(f"   {float(m2):.15f}")
    print("maxt2 =")
    print(f"   {float(p2):.15f}")


if __name__ == "__main__":
    run()
