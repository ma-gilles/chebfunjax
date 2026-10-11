"""The Carrier equation.

Translation of ode-nonlin/Carrier.m by Nick Trefethen and Asgeir
Birkisson (October 2010): the Carrier boundary-layer problem

    0.01 u'' + 2(1-x^2) u + u^2 = 1,   u(-1) = u(1) = 0,

which has many solutions — the one Newton finds depends on the initial
guess — together with the Newton convergence history and a variant
with a Robin condition at the right end.

Original: https://www.chebfun.org/examples/ode-nonlin/Carrier.html
Copyright by The University of Oxford and The Chebfun Developers.
"""
import matplotlib

matplotlib.use("Agg")
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators.chebop import Chebop
from chebfunjax.plotting import _source_plot_line_and_points, chebfun_style, matlab_plot
from chebfunjax.plotting import save_chebfun_figure as _savefig

chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, '..', '..', 'docs', 'images', 'ode-nonlin')

FIG = [0]


def _output_slot(number):
    if os.environ.get("CHEBFUNJAX_OUTPUT_SLOTS") == "1":
        print(f"@@CHEBFUNJAX_OUTPUT_SLOT:{number}@@", flush=True)


def _capture_plain(value):
    if isinstance(value, dict):
        return {str(k): _capture_plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_capture_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _capture_plain(value.item())
    if hasattr(value, "dtype"):
        return _capture_plain(np.asarray(value).tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def _capture_solution(u, nrmdu, info, fig, axs):
    root = os.environ.get("CHEBFUNJAX_CAPTURE_DIR")
    if not root:
        return
    os.makedirs(root, exist_ok=True)
    arrays = {"domain": np.asarray(tuple(u.domain)), "nrmdu": np.asarray(nrmdu),
              "point_values": np.asarray(u.point_values)}
    stored = getattr(u, "_point_values", None)
    if stored is not None:
        arrays["stored_point_values"] = np.asarray(stored)
    for k, piece in enumerate(u.funs):
        arrays[f"coeffs_{k}"] = np.asarray(piece.tech.coeffs)
        arrays[f"interval_{k}"] = np.asarray(piece.interval)
    for a, ax in enumerate(axs):
        for k, artist in enumerate(ax.lines):
            arrays[f"artist_{a}_{k}_x"] = np.asarray(artist.get_xdata())
            arrays[f"artist_{a}_{k}_y"] = np.asarray(artist.get_ydata())
    stem = os.path.join(root, f"Carrier_{FIG[0]:02d}")
    np.savez(stem + ".npz", **arrays)
    metadata = {"point_values_override": stored is not None,
                "is_transposed": bool(u.is_transposed), "info": info,
                "repr": repr(u), "fig_inches": fig.get_size_inches(),
                "axes": [{"position": ax.get_position().bounds,
                          "xlim": ax.get_xlim(), "ylim": ax.get_ylim(),
                          "yscale": ax.get_yscale()} for ax in axs]}
    with open(stem + ".json", "w") as handle:
        json.dump(_capture_plain(metadata), handle, indent=2)


def _carrier_page_layout(axs, count):
    """Carrier cached-page subplot geometry/limits; not general autoscaling."""
    for ax, left in zip(axs, (0.13, 0.570340909090909)):
        ax.set_position((left, 0.11, 0.334659090909091, 0.815))
        ax.tick_params(direction="in", top=True, right=True, labelsize=10)
    axs[0].set_xlim(-1, 1)
    axs[0].set_xticks(np.arange(-1, 1.1, .5))
    axs[0].set_ylim((-2.5, .5) if count == 6 else (-2, 2))
    axs[0].set_yticks(np.arange(-2.5, .6, .5) if count == 6 else np.arange(-2, 2.1, .5))
    axs[0].grid(False)
    axs[1].set_xlim(1, count)
    axs[1].set_xticks(np.arange(1, 7) if count == 6 else np.arange(2, count + 1, 2))
    axs[1].set_ylim((1e-15, 1) if count == 6 else (1e-15, 1e5))
    axs[1].set_yticks([1e-15, 1e-10, 1e-5, 1] + ([] if count == 6 else [1e5]))
    axs[1].grid(True, which="major", color="#d9d9d9", linewidth=.6)
    axs[1].grid(False, which="minor")


def _save(u, nrmdu, *, info=None):
    FIG[0] += 1
    fig, axs = plt.subplots(1, 2, figsize=(600 / 72, 270 / 72), dpi=72)
    matlab_plot(u, ax=axs[0], lw=1.6)
    axs[0].set_title("Solution", fontsize=14, fontweight="bold")
    axs[1].set_yscale("log")
    steps = np.arange(1, len(nrmdu) + 1)
    _source_plot_line_and_points(axs[1], steps, nrmdu, steps, nrmdu,
                                 '.-r', {'linewidth': 1.6, 'markersize': 16})
    axs[1].set_title("Convergence", fontsize=14, fontweight="bold")
    axs[1].set_xlim(1, len(nrmdu))
    axs[1].grid(True)
    fig.set_facecolor("white")
    _carrier_page_layout(axs, len(nrmdu))
    _capture_solution(u, nrmdu, info, fig, axs)
    _savefig(fig, os.path.join(_IMG, f"Carrier_{FIG[0]:02d}.png"), dpi=72)
    plt.close(fig)


def _show_accuracy(N, u):
    print("accuracy =")
    print(f"     {float((N(u) - 1).norm()):.15e}")


def run():
    os.makedirs(_IMG, exist_ok=True)

    N = Chebop(lambda x, u: 0.01 * u.diff(2) + 2 * (1 - x ** 2) * u + u ** 2,
               domain=(-1, 1))
    N.bc = 'dirichlet'
    x = cj.chebfun('x')
    N.init = 2 * (x ** 2 - 1)

    ChebopPref.setDefaults("display", "iter")
    _output_slot(1)
    u, info = N.solvebvp(1)
    nrmdu = info["normDelta"]
    _save(u, nrmdu, info=info)
    _output_slot(2)
    _show_accuracy(N, u)

    N.init = 2 * (x ** 2 - 1) * (1 - 2 / (1 + 20 * x ** 2))
    ChebopPref.setDefaults("display", "off")
    u, info = N.solvebvp(1)
    nrmdu = info["normDelta"]
    _save(u, nrmdu, info=info)
    _output_slot(3)
    _show_accuracy(N, u)

    N.lbc = 1
    N.rbc = lambda u: u.diff() + u
    u, info = N.solvebvp(1)
    nrmdu = info["normDelta"]
    _save(u, nrmdu, info=info)
    _output_slot(4)
    _show_accuracy(N, u)


if __name__ == "__main__":
    run()
