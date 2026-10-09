"""Sampling from probability distributions by inverting their CDFs.

Translation of stats/ResamplingRandomVariables.m by Toby Driscoll
(December 2011). The Chebfun computations use the public ChebfunJAX
constructor, sum, cumsum, restrict, inv and plotting APIs. NumPy's
MT19937 stream is retained as a deterministic Python sampling adapter;
MATLAB rand stream equivalence is not established.

Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Example commit: f4b9ea46cfc2f52f20a844627f4a74d0bb10098c

Original: https://www.chebfun.org/examples/stats/ResamplingRandomVariables.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import os
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# uses-numpy: deterministic RNG and MATLAB histogram/display host adapters; inverse math uses Chebfuns.
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig
from chebfunjax.utils.matlab_hist import matlab_hist_counts

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, "..", "..", "docs", "images", "stats")
_FIG = 0


def _save(fig):
    global _FIG
    _FIG += 1
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(
        _IMG, f"ResamplingRandomVariables_{_FIG:02d}.png"), size=(600, 270))
    plt.close(fig)


def _hist36(values):
    """Source scalar-bin histogram and area normalization through JAX."""
    counts, centers = matlab_hist_counts(jnp.asarray(values), 36)
    width = centers[1] - centers[0]
    counts = counts / jnp.sum(counts * width)
    return counts, centers, width


def _plot_histogram(values, density, xlim=None):
    counts, centers, width = _hist36(values)
    fig, ax = plt.subplots()
    # MATLAB bar's default width occupies 0.8 of each bin spacing.
    ax.bar(centers, counts, width=0.8 * width, color="#352A86",
           edgecolor="black", linewidth=0.5)
    matlab_plot(density, "r", ax=ax, lw=1.6)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.autoscale(tight=True)
    return fig


def _plot_columns(*columns, ylim=None):
    fig, ax = plt.subplots()
    # A list of scalar Chebfuns is the Python quasimatrix representation.
    matlab_plot(cj.cell2quasi(list(columns)), ax=ax, lw=1.6)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax


def _ans_vec(values, name="ans"):
    print(f"{name} =")
    for value in values:
        print(f"{float(value):20.15f}")


def run():
    os.makedirs(_IMG, exist_ok=True)
    rng = np.random.RandomState(5489)  # Python MT19937 adapter; not MATLAB-stream parity.

    # von Mises distribution. MATLAB source: f / sum(f).
    kappa = 1.5
    f = cj.chebfun(
        lambda x: jnp.exp(kappa * jnp.cos(x)),
        domain=(-np.pi, np.pi), splitting=False)
    density = f / f.sum()
    cdf = density.cumsum()

    fig, ax = _plot_columns(density, cdf, ylim=(0, 1))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_title("von Mises distribution", fontsize=12)
    ax.legend(["density", "distribution"], loc="upper left")
    _save(fig)

    cdfinv = cdf.inv()
    fig, ax = plt.subplots()
    matlab_plot(cdfinv, ax=ax, lw=1.6)
    ax.set_title("Inverse of von Mises distribution", fontsize=12)
    _save(fig)

    u = rng.rand(10**4)
    x = np.asarray(cdfinv(jnp.asarray(u)))
    fig = _plot_histogram(x, density)
    fig.axes[0].set_title(
        "Sampled points and the orignal density", fontsize=12)
    _save(fig)

    # Logit-normal density and CDF. The source constructs on [0,1] and
    # relies on the endpoint limit; no epsilon-trimmed surrogate is used.
    sig = 1.11

    def logit_normal_density(x):
        return (jnp.exp(-(jnp.log(x / (1 - x))) ** 2 / (2 * sig**2))
                / (x * (1 - x)))

    density2 = cj.chebfun(
        logit_normal_density, domain=(0.0, 1.0), splitting=False)
    density2 = density2 / density2.sum()
    cdf2 = density2.cumsum()

    fig, ax = _plot_columns(density2, cdf2)
    ax.set_title("logit-normal distribution", fontsize=12)
    ax.legend(["density", "distribution"], loc="upper left")
    _save(fig)

    # MATLAB cdf{a,b} calls restrict then simplify (@chebfun/subsref.m).
    # Preserve that source operation using the public ChebfunJAX APIs.
    cdfinv2 = (cdf2.restrict(0.5, 1 - 1e-3).simplify()
               .inv(splitting=True))
    fig, ax = plt.subplots()
    matlab_plot(cdfinv2, ax=ax, lw=1.6)
    ax.set_title("Inverse of the logit-normal distribution", fontsize=12)
    _save(fig)

    u = rng.rand(10**4)
    flag = u < 0.5
    u[flag] = 1 - u[flag]
    x = cdfinv2(jnp.asarray(u))
    x = jnp.where(jnp.asarray(flag), 1 - x, x)
    fig = _plot_histogram(x, density2)
    fig.axes[0].set_title(
        "Sampled points and the orignal density", fontsize=12)
    _save(fig)

    # MATLAB cdfinv.ends.' prints the complete breakpoint vector.
    ends = np.asarray(cdfinv2.domain.breakpoints, dtype=float)
    _ans_vec(ends)
    missing = 1.0 - float(ends[-1])
    print("missing =")
    print(f"     {missing:.15e}")


if __name__ == "__main__":
    run()
