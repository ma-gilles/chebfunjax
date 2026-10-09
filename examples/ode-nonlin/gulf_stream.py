"""A Gulf Stream model.

Translation of ode-nonlin/GulfStream.m by C. I. Gheorghiu
(January 2020): the third-order nonlinear
boundary-value problem

    u''' - lambda ((u')^2 - u u'') - u + 1 = 0,  x in [0, 35],

with stress-free conditions u(0) = u''(0) = 0 and u(35) = 1, plus the
conserved quantity I = int (u'')^2 - 3 lambda u u' u'' = 1/2.

Original: https://www.chebfun.org/examples/ode-nonlin/GulfStream.html
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import chebfun_style, matlab_plot, plotcoeffs
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]
X = 35.0
LAM = -0.1


def _source_style(fig, slot):
    from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
    fig.set_facecolor("white")
    fig.set_size_inches(6, 2.53)
    ax = fig.axes[0]
    ax.set_position([.13, .15, .775, .76])
    ax.tick_params(direction="in", top=True, right=True)
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if slot == 1:
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    elif slot == 2:
        ax.set_yticks([1, 1e-10])
        ax.set_xlim(1, len(ax.lines[0].get_xdata()))
    else:
        for line in ax.lines:
            line.set_markersize(3.5)
            line.set_markeredgewidth(.5)
        ax.set_yticks([1, 1e-5, 1e-10, 1e-15])
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1,), numticks=30))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=".85", linewidth=.5)
        ax.grid(True, which="minor", color=".85", linewidth=.5, linestyle=":")


def _save(fig):
    FIG[0] += 1
    _source_style(fig, FIG[0])
    _savefig(fig, os.path.join(
        _IMG, f"GulfStream_{FIG[0]:02d}.png"), size=(600, 253))
    plt.close(fig)


def _op(u):
    return (u.diff(3) - LAM * (u.diff(1)**2 - u * u.diff(2))
            - u + 1)


def run():
    os.makedirs(_IMG, exist_ok=True)
    t0 = time.time()

    N = Chebop(_op, domain=(0, X))
    N.lbc = lambda u: [u, u.diff(2)]       # stress-free
    N.rbc = 1
    u, info = N.solvebvp(0.0)

    fig, ax = plt.subplots(figsize=(6, 2.53))
    ax.set_prop_cycle(color=["#0072BD", "#D95319", "#EDB120"])
    matlab_plot([u, u.diff(), u.diff(2)], ax=ax, linewidth=1.5)
    ax.axis([0, 20, -1, 1.5])
    ax.set_xlabel("x")
    ax.legend(["u", "u'", "u''"], loc="lower right", fontsize=9,
              labelspacing=.1, borderpad=.3, fancybox=False, edgecolor=".4")
    ax.set_title("Slippery or stress-free b. c.")
    _save(fig)

    print("N_residual =")
    print(f"     {float(_op(u).norm()):.15e}")
    print("lbc_residuals =")
    print(f"   {float(u(jnp.array(0.0))):.15e}  "
          f"{float(u.diff(2)(jnp.array(0.0))):.15e}")
    print("rbc_residual =")
    print(f"    {float(u(jnp.array(X))) - 1:.15e}")

    nd = info["normDelta"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.semilogy(np.arange(1, len(nd) + 1), nd, 'm*-', lw=1.5)
    ax.set_ylim(1e-16, 1e1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("norm of Newton update")
    _save(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    plotcoeffs(u, ax=ax, source=True, color="#0072BD")
    _save(fig)

    I = float((u.diff(2)**2
               - 3 * LAM * (u * u.diff() * u.diff(2))).sum())
    print("I =")
    print(f"   {I:.15f}")
    print("I_error =")
    print(f"     {abs(I - 0.5):.15e}")
    print("total_time_for_this_example =")
    print(f"   {time.time() - t0:.15f}")


if __name__ == "__main__":
    run()
