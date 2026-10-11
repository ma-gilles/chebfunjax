"""Gauss and Clenshaw-Curtis quadrature.

Translation of quad/GaussClenCurt.m by Nick Trefethen: for a
wiggly function, chebfun's sum, Clenshaw-Curtis at the chebfun length,
and Gauss quadrature all agree; the two rules' convergence curves are
compared.

Original: https://www.chebfun.org/examples/quad/GaussClenCurt.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import os
import sys
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.quadrature import chebpts, chebweights, legpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'quad')


def _publish(fig, name):
    """Apply the cached page's export geometry to the computed plot.

    MATLAB font/marker points and Matplotlib export points have different
    visible sizes here. These are page rendering choices; numerical arrays
    and the source plotting calls below retain their original values.
    Historical font rasterization and exact pixel parity remain unqualified.
    """
    ax = fig.axes[0]
    ax.set_position([.13, .11, .775, .815])
    ax.title.set_fontsize(11)
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(8.5)
    ax.yaxis.label.set_fontsize(8.5)
    ax.tick_params(labelsize=8, top=True, right=True)
    for line in ax.lines:
        line.set_linewidth(.9 if name.endswith('01.png') else .75)
        if line.get_marker() == '.':
            line.set_markersize(4)
    if not name.endswith('01.png'):
        ax.set_xlim(0, 500)
        ax.set_xticks(range(0, 501, 50))
        ax.set_yscale('log', nonpositive='mask')
        ax.set_yticks([1, 1e-5, 1e-10, 1e-15])
        ax.grid(True, color='#b0b0b0', alpha=.35, linewidth=.5)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(8)
                text.set_fontfamily('Arial')
            for line in legend.get_lines():
                line.set_markersize(4)
                line.set_linewidth(.75)
    for text in [ax.title, ax.xaxis.label, ax.yaxis.label,
                 *ax.get_xticklabels(), *ax.get_yticklabels()]:
        text.set_fontfamily('Arial')
    _savefig(fig, os.path.join(_IMG, name), size=(600, 270))


def run():
    os.makedirs(_IMG, exist_ok=True)
    f = lambda x: x * jnp.sin(2 * jnp.exp(2 * jnp.sin(2 * jnp.exp(2 * x))))
    fc = cj.chebfun(f)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    matlab_plot(fc, ax=ax, linewidth=1.2)
    ax.set_title("Function f", fontsize=16)
    fig.set_facecolor("white")
    fig.tight_layout()
    _publish(fig, "GaussClenCurt_01.png")
    plt.close(fig)

    Icheb = float(fc.sum())
    print("Ichebfun =")
    print(f"   {Icheb:.15f}")
    N = len(fc)
    print("Npts =")
    print(f"   {N}")
    s = chebpts(N)
    w = chebweights(N)
    print("Iclenshawcurtis =")
    print(f"   {float(w @ f(s)):.15f}")
    sg, wg = legpts(N)
    print("Igauss =")
    print(f"   {float(wg @ f(sg)):.15f}")

    # Source has two successive figures: Gauss alone, then both rules.
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    tic = time.perf_counter()
    NN = range(10, 501, 10)
    errG = []
    for Npts in NN:
        sg, wg = legpts(Npts)
        errG.append(abs(float(wg @ f(sg)) - Icheb))
    ax.semilogy(NN, errG, ".-", lw=1, ms=16, label="Gauss")
    ax.set_ylim(1e-18, 1)
    ax.grid(True)
    ax.set_xlabel("Npts", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Gauss quadrature convergence", fontsize=16)
    print(f"Elapsed time is {time.perf_counter() - tic:.6f} seconds.")
    fig.set_facecolor("white")
    fig.tight_layout()
    _publish(fig, "GaussClenCurt_02.png")

    tic = time.perf_counter()
    errC = []
    for Npts in NN:
        s = chebpts(Npts)
        w = chebweights(Npts)
        errC.append(abs(float(w @ f(s)) - Icheb))
    ax.semilogy(NN, errC, ".-r", lw=1, ms=16, label="Clenshaw-Curtis")
    ax.set_title("Gauss and Clenshaw-Curtis", fontsize=16)
    ax.legend(loc="lower left")
    print(f"Elapsed time is {time.perf_counter() - tic:.6f} seconds.")
    _publish(fig, "GaussClenCurt_03.png")
    plt.close(fig)
    return True


if __name__ == "__main__":
    run()
