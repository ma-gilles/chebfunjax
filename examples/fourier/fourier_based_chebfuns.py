"""Fourier-based chebfuns.

Translation of fourier/FourierBasedChebfuns.m by Grady Wright, June 2014.
The cached page is the reference for prose, output and figure audits.

Original: https://www.chebfun.org/examples/fourier/FourierBasedChebfuns.html
Copyright 2014 by The University of Oxford and The Chebfun Developers.

Exact MATLAB Gaussian-stream parity and rendered figure parity remain open.
Printed values are computed by the current library rather than substituted
from the reference page.
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
from chebfunjax.plotting import chebfun_style, matlab_axis_equal, plotcoeffs, trig_plot_data
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.quadrature import trigpts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'fourier')


def _show(name, f):
    print(f"{name} =")
    print(repr(f))


def _source_style(fig, stem):
    from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
    ax = fig.axes[0]
    slot = int(stem.rsplit("_", 1)[1])
    ax.tick_params(direction="in", top=True, right=True)
    if slot != 2:
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    if slot in (1, 3, 4, 5, 6, 7):
        ax.lines[0].set_color((0, .447, .741))
        ax.lines[0].set_linewidth(1.5)
    if slot == 8:
        for line in ax.lines:
            line.set_linewidth(1.5)
    ticks = {1: [-1, -.5, 0, .5, 1], 3: [0, .2, .4, .6, .8, 1], 6: [-2, -1, 0, 1, 2], 8: [0, 1, 2, 3, 4]}
    if slot in ticks:
        ax.set_yticks(ticks[slot])
    if slot == 2:
        ax.set_position([.13, .16, .775, .75])
        ax.title.set_fontsize(12)
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        for line in ax.lines:
            line.set_markersize(2.5)
        ax.set_yticks([1, 1e-5, 1e-10, 1e-15])
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1,), numticks=30))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=".85", linewidth=.5)
        ax.grid(True, which="minor", color=".85", linewidth=.5, linestyle=":")
    if slot == 5:
        for line, color in zip(ax.lines[1:3], [(0, 1, 0), (1, 0, 1)]):
            line.set_color(color)
        for line in ax.lines[1:]:
            line.set_markersize(7)
    legend = ax.get_legend()
    if legend is not None:
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc="lower left" if slot == 5 else "upper right",
                           fontsize=9, labelspacing=.15, borderpad=.3, handletextpad=.3,
                           handlelength=3.2 if slot == 8 else 2)
        for handle, line in zip(legend.legend_handles, ax.lines):
            handle.set_color(line.get_color())
            handle.set_linewidth(line.get_linewidth())
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(.5)
        legend.get_frame().set_alpha(1)
        legend.get_frame().set_boxstyle("square", pad=0)


def _save(fig, stem):
    fig.set_facecolor("white")
    fig.set_size_inches(6, 2.7)
    fig.axes[0].set_position([.13, .11, .775, .815])
    _source_style(fig, stem)
    _savefig(fig, os.path.join(_IMG, stem + ".png"))
    plt.close(fig)


def _plot(ax, f, style, *, complex_curve=False, **kwargs):
    data = trig_plot_data(f)
    ax.plot(np.asarray(data["xLine"]), np.asarray(data["yLine"]), style, **kwargs)
    ax.set_position([.13, .11, .775, .815])
    if not complex_curve:
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks(np.arange(-3, 4))


def run():
    os.makedirs(_IMG, exist_ok=True)
    dom = [-np.pi, np.pi]

    # -- Construction and comparison --------------------------------
    f = cj.chebfun(lambda x: jnp.cos(8 * jnp.sin(x)), domain=dom,
                   trig=True)
    _show("f", f)
    fig, ax = plt.subplots(figsize=(6, 2.7))
    _plot(ax, f, "b")
    ax.set_ylim(-1, 1)
    _save(fig, "FourierBasedChebfuns_01")

    fig, ax = plt.subplots(figsize=(6, 2.7))
    plotcoeffs(f, ax=ax, source=True)
    ax.set_ylim(1e-18, 1)
    _save(fig, "FourierBasedChebfuns_02")

    f_cheby = cj.chebfun(lambda x: jnp.cos(8 * jnp.sin(x)), domain=dom)
    _show("f_cheby", f_cheby)

    print("ratio =")
    print(f"   {len(f_cheby) / len(f):.15f}")
    print("theoretical =")
    print(f"   {np.pi / 2:.15f}")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        f_step = cj.chebfun(
            lambda x: 0.5 * (1.0 + jnp.sign(x)), domain=dom, trig=True)
    for warning in rec:
        message = str(warning.message)
        expected = (f"Trigtech.from_function: function did not converge with "
                    f"{len(f_step)} points. Returning unhappy representation.")
        if message == expected and not f_step.funs[0].tech.ishappy:
            # Render this actual unresolved-construction warning in MATLAB's
            # published display format; preserve any other warning verbatim.
            print(f"Warning: Function not resolved using {len(f_step)} pts. "
                  "Have you tried a non-trig")
            print("representation? ")
        else:
            print("Warning:", message)
    _show("f", f_step)
    fig, ax = plt.subplots(figsize=(6, 2.7))
    _plot(ax, f_step, "b")
    ax.set_ylim(0, 1)
    _save(fig, "FourierBasedChebfuns_03")

    f_split = cj.chebfun(lambda x: 0.5 * (1.0 + jnp.sign(x)),
                         domain=dom, splitting=True)
    _show("f", f_split)

    # -- Basic operations -------------------------------------------
    g = cj.chebfun(lambda x: jnp.sin(x), domain=dom, trig=True)
    f = ((1 + 2 * g).cos() ** 2).tanh() - 0.5
    _show("f", f)
    fig, ax = plt.subplots(figsize=(6, 2.7))
    _plot(ax, f, "b")
    ax.set_ylim(-.6, .4)
    _save(fig, "FourierBasedChebfuns_04")

    (xminf, minf), (xmaxf, maxf) = f.minandmax()
    rootsf = np.sort(np.asarray(f.roots()))
    print("maxf =")
    print(f"   {float(maxf):.15f}")
    print("minf =")
    print(f"  {float(minf):.15f}")
    print("rootsf =")
    for r in rootsf:
        print(f"  {r: .15f}")

    fig, ax = plt.subplots(figsize=(6, 2.7))
    _plot(ax, f, "b", label="f")
    ax.set_ylim(-.6, .4)
    ax.plot([xmaxf], [maxf], "gs", markerfacecolor="none", label="max f")
    ax.plot([xminf], [minf], "md", markerfacecolor="none", label="min f")
    ax.plot(rootsf, 0 * rootsf, "ro", markerfacecolor="none", label="zeros f")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, "FourierBasedChebfuns_05")

    df = f.diff()
    fig, ax = plt.subplots(figsize=(6, 2.7))
    _plot(ax, df, "b")
    ax.set_ylim(-2, 2)
    _save(fig, "FourierBasedChebfuns_06")

    print("intf =")
    print(f"  {float(f.sum()):.15f}")

    # -- Complex-valued trigfuns: the heart curve --------------------
    fh = cj.chebfun(
        lambda x: 1j * (13 * jnp.cos(x) - 5 * jnp.cos(2 * x)
                        - 2 * jnp.cos(3 * x) - jnp.cos(4 * x))
        + 16 * jnp.sin(x) ** 3, domain=dom, trig=True)
    _show("f", fh)
    fig, ax = plt.subplots(figsize=(6, 2.7))
    _plot(ax, fh, "b", complex_curve=True)
    ax.set_ylim(-17, 12)
    matlab_axis_equal(ax)
    _save(fig, "FourierBasedChebfuns_07")

    area_heart = abs(float((fh.real() * fh.imag().diff()).sum()))
    print("area_heart =")
    print(f"     {area_heart:.15e}")
    err = (area_heart - 180 * np.pi) / (180 * np.pi)
    print("err =")
    print(("     " if err > 0 else "    ") + f"{err:.15e}" if err
          else "     0")

    # -- circconv + construction from values -------------------------
    rng = np.random.RandomState(0)
    n = 201
    x, _ = trigpts(n)  # source samples on default [-1,1), then uses dom
    # NumPy seed 0 is reproducible here; native MATLAB Gaussian draws have
    # not been captured, so this realization is not a native RNG fixture.
    func_vals = jnp.exp(jnp.sin(2 * jnp.pi * x)) + 0.05 * jnp.asarray(rng.randn(n))
    fN = cj.chebfun(func_vals, domain=dom, trig=True)
    _show("f", fN)

    sigma = 0.1
    gm = cj.chebfun(
        lambda t: 1 / (sigma * np.sqrt(2 * np.pi))
        * jnp.exp(-0.5 * (t / sigma) ** 2), domain=dom, trig=True)
    h = fN.circconv(gm)
    fig, ax = plt.subplots(figsize=(6, 2.7))
    _plot(ax, gm, "b", label="Mollifier g")
    _plot(ax, fN, "r", label="Noisy function f")
    _plot(ax, h, "k", label="Smoothed function h")
    ax.set_ylim(0, 4)
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "FourierBasedChebfuns_08")

    return True


if __name__ == "__main__":
    run()
