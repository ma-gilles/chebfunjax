"""The checkmark function.

Translation of approx/Checkmark.m by Nick Trefethen (September 2019):
the error E_n(alpha) of the best degree-n polynomial approximation to
|x - alpha| on [-1, 1] as a function of alpha, whose graph for n = 3 has
a local minimum at alpha = 0 (a "checkmark").

Original: https://www.chebfun.org/examples/approx/Checkmark.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj  # noqa: E402
from chebfunjax.plotting import chebfun_style  # noqa: E402
from chebfunjax.plotting import save_chebfun_figure as _savefig  # noqa: E402
from chebfunjax.utils.minimax import minimax  # noqa: E402

jax.config.update("jax_enable_x64", True)
chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'approx')

RED = (0.9, 0, 0)
BLUE = (0, 0, 0.9)


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")
    t0 = time.time()

    # x = chebfun('x'); f = @(a) abs(x-a); p = @(a,n) minimax(f(a),n);
    # e = @(a,n) norm(p(a,n)-f(a),inf);
    def f(a):
        # abs(x - a) introduces a breakpoint at the interior root a only.
        dom = (-1.0, a, 1.0) if -1.0 < a < 1.0 else (-1.0, 1.0)
        return cj.chebfun(lambda t: jnp.abs(t - a), domain=dom)

    def p(a, n):
        bp = [float(a)] if -1.0 < a < 1.0 else None
        return minimax(f(a), n, breakpoints=bp)

    def e(a, n):
        # norm(p(a,n) - f(a), inf): the Remez error of the converged best
        # approximation (equal to the sup norm of p - f to rounding).
        return float(p(a, n).err)

    # chebfuneps 1e-6, splitting on
    E = []
    for n in range(1, 8):
        def en_vals(a, _n=n):
            arr = np.atleast_1d(np.asarray(a, dtype=np.float64))
            out = [e(float(v), _n) for v in arr.ravel()]
            return jnp.asarray(out, dtype=jnp.float64).reshape(arr.shape)
        en = cj.chebfun(en_vals, domain=(0.0, 1.0), eps=1e-6, splitting=True)
        en = en.flipud().join(en).new_domain((-1.0, 1.0))
        E.append(en)
    # chebfuneps factory, splitting off

    al = np.linspace(-1, 1, 2001)

    fig, ax = plt.subplots(figsize=(6, 2.7))
    for en, col in ((E[1], RED), (E[2], BLUE)):
        ax.plot(al, np.asarray(en(jnp.asarray(al))), color=col, lw=1.0)
    ax.grid(True)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$E_n(\alpha)$")
    ax.set_title("n = 2 and 3", fontsize=12)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Checkmark_01.png"), size=(600, 270))
    plt.close(fig)

    # [val,pos] = min(E(:,3),'local')
    pos, val = E[2].min("local")
    print("val =")
    for v in np.asarray(val).ravel():
        print(f"   {v:.15f}")
    print("pos =")
    for x in np.asarray(pos).ravel():
        print(f"  {x:.15f}" if x != 0 else "                   0")

    fig, ax = plt.subplots(figsize=(6, 2.7))
    for k, en in enumerate(E, start=1):
        ax.plot(al, np.asarray(en(jnp.asarray(al))),
                color=BLUE if k % 2 == 1 else RED, lw=1.0)
    ax.grid(True)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$E_n(\alpha)$")
    ax.set_title("n = 1,2,...,7", fontsize=12)
    ax.set_ylim(0, 0.5)
    fig.set_facecolor("white")
    fig.tight_layout()
    _savefig(fig, os.path.join(_IMG, "Checkmark_02.png"), size=(600, 270))
    plt.close(fig)

    print(f"Elapsed time is {time.time() - t0:.6f} seconds.")


if __name__ == "__main__":
    run()
