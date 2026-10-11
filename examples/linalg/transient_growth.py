"""Transient growth in linear systems.

Translation of linalg/TransientGrowth.m by Nick Trefethen
(May 2011, after a laser-physics example of Kestutis Staliunas): the
norm and energy of exp(tA) for a stable 7x7 matrix showing enormous
transient amplification before eventual decay.

Original: https://www.chebfun.org/examples/linalg/TransientGrowth.html
Copyright by The University of Oxford and The Chebfun Developers.
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
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'linalg')

A = np.array([
    [-1, 0, 0, 0, 0, 0, -625],
    [0, -1, -30, 400, 0, 0, 250],
    [-2, 0, -1, 0, 0, 0, 30],
    [5, -1, 5, -1, 0, 0, 200],
    [11, 1, 25, -10, -1, 1, -200],
    [200, 0, 0, -150, -100, -1, -1000],
    [1, 0, 0, 0, 0, 0, -1]], dtype=float)


def _source_transient_figure(index, f):
    """Source plot with observed R2024b 600x270 axes geometry."""
    fig, ax = plt.subplots(figsize=(600 / 72, 270 / 72), dpi=72)
    position = (0.13, 0.12851851851851853, 0.775, 0.7853703703703705)
    ax.set_position(position)
    matlab_plot(f, "b", linewidth=2, ax=ax)
    ax.set_xlim(0, 2.5)
    ax.set_xticks((0, 0.5, 1, 1.5, 2, 2.5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    if index == 1:
        ax.set_ylim(0, 600)
        ax.set_yticks(np.arange(0, 601, 100))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ylabel, title = r"$\|e^{tA}\|$", "amplitude"
    else:
        ax.set_ylim(0, 400000)
        ax.set_yticks(np.arange(0, 400001, 50000))
        # Observed native YAxis.Exponent is 5, with labels 0,.5,1,...,4.
        formatter = FuncFormatter(lambda value, _: f"{value / 1e5:g}")
        formatter.get_offset = lambda: r"$\times 10^5$"
        ax.yaxis.set_major_formatter(formatter)
        ylabel, title = r"$\|e^{tA}\|^2$", "energy"
    tick_length = 0.01 * max(600 * position[2], 270 * position[3])
    ax.tick_params(labelsize=10, direction="in", top=True, right=True,
                   width=0.5, length=tick_length, colors=(0.15, 0.15, 0.15))
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color((0.15, 0.15, 0.15))
    ax.set_xlabel("t", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, f"TransientGrowth_{index:02d}.png"),
             size=(600, 270), dpi=72)
    plt.close(fig)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()

    # MATLAB's display of A (format long, 7 columns wrap at 6).
    print("A =")
    for j0, j1, head in ((0, 6, "  Columns 1 through 6"),
                         (6, 7, "  Column 7")):
        print(head)
        for row in A:
            print("".join(f"{int(v):12d}" for v in row[j0:j1]))

    def op(t_arr):
        t_arr = np.atleast_1d(np.asarray(t_arr, dtype=float))
        out = np.empty_like(t_arr)
        for i, t in enumerate(t_arr.ravel()):
            out.ravel()[i] = np.linalg.norm(sla.expm(t * A), 2)
        return out.reshape(np.shape(t_arr))

    e = cj.chebfun(lambda t: jnp.asarray(op(np.asarray(t))),
                   domain=(0.0, 2.5), splitting=True)
    _source_transient_figure(1, e)

    e2 = e ** 2
    _source_transient_figure(2, e2)

    _, m2 = e2.max()
    print(f"Maximum energy = {float(m2):15.8f}")
    print(f"Elapsed time is {time.time()-t0:.6f} seconds.")


if __name__ == "__main__":
    run()
