"""The Lane-Emden equation.

Translation of ode-nonlin/LaneEmden.m (Chebfun example): solutions of
the Lane-Emden equation x u'' + 2 u' + x u^n = 0, u(0) = 1, u'(0) = 0,
for n = 0, ..., 5, a check against the closed form for n = 5, and the
white-dwarf problem posed as a boundary-value problem for u and the
unknown radius v.

Original: https://www.chebfun.org/examples/ode-nonlin/LaneEmden.html
Copyright by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import os
import sys
import warnings

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import chebfunjax as cj  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402
from chebfunjax.plotting import chebfun_style, matlab_plot  # noqa: E402
from chebfunjax.plotting import save_chebfun_figure as _savefig  # noqa: E402

jax.config.update("jax_enable_x64", True)
chebfun_style()
_HERE = os.path.dirname(os.path.abspath(__file__))
_IMG = os.path.join(_HERE, "..", "..", "docs", "images", "ode-nonlin")


def _real_display_solution(u, radius):
    """Display the real physical branch, rejecting non-roundoff imaginary data.

    The MATLAB page's plot/axis/fprintf calls assume a real solution. This
    visualization adapter preserves the solver's complex result and only
    projects values bounded by100 binary64 eps times their own scale. The
    coefficient L1 bound controls the entire polynomial curve, not a grid.
    This threshold does not test Newton convergence or the ODE residual.
    """
    rounding = 100 * np.finfo(float).eps
    radius_value = complex(radius)
    if abs(radius_value.imag) > rounding * max(1., abs(radius_value.real)):
        raise ValueError("white-dwarf radius is materially complex; cannot display real branch")
    imaginary_bound = max(float(np.sum(np.abs(np.imag(np.asarray(piece.tech.coeffs)))))
                          for piece in u.funs)
    imaginary_bound = max(imaginary_bound,
                          float(np.max(np.abs(np.imag(np.asarray(u(u.domain.breakpoints)))))))
    if imaginary_bound > rounding * max(1., float(u.vscale)):
        raise ValueError("white-dwarf solution is materially complex; cannot display real branch")
    return u.real(), radius_value.real


def run():
    os.makedirs(_IMG, exist_ok=True)
    warnings.filterwarnings("ignore")

    fig, ax = plt.subplots(figsize=(600 / 72.009, 269 / 72.009))
    ax.set_position([0.13, 0.14, 0.775, 0.775])
    ax.tick_params(labelsize=12)
    for n in range(6):
        # N = chebop(@(x,u) x*diff(u,2) + 2*diff(u) + x*u^n, [0,10]);
        N = Chebop(lambda x, u, _n=n: x * u.diff(2) + 2 * u.diff() + x * u**_n, domain=(0.0, 10.0))
        N.lbc = lambda u: [u - 1, u.diff()]
        u = N.solvebvp(0.0)[0]
        matlab_plot(u, ax=ax, linewidth=2, label=f"n={n}")
    ax.axis([0, 10, -1, 1])
    ax.set_title("Solution of the Lane-Emden equation for n=0,1,2,3,4,5", fontsize=14)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u", fontsize=14)
    ax.legend(
        loc="upper right",
        fontsize=12,
        handlelength=1,
        handletextpad=0.3,
        borderpad=0.2,
        framealpha=1,
    )
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, "LaneEmden_01.png"), size=(600, 269), dpi=72.009)
    plt.close(fig)

    f = cj.chebfun(lambda x: 1 / jnp.sqrt(1 + x**2 / 3), domain=(0.0, 10.0))
    print(f"The L2 error is: {float((f - u).norm()):1.3e}")

    # White dwarfs: rescale to [0, 1] with the unknown radius v.
    d = (0.0, 1.0)
    x = cj.chebfun(lambda t: t, domain=d)
    N = Chebop(domain=d)
    n = 1.5
    N.op = lambda x, u, v: x * u.diff(2) + 2 * u.diff() + x * v**2 * u**n
    N.lbc = lambda u, v: [u - 1, u.diff()]
    N.rbc = lambda u, v: u
    N.init = [(np.pi / 2 * x).cos(), np.pi]
    uv = N.solvebvp(0.0)[0]
    u, v = uv[0], uv[1]
    u_display, v0 = _real_display_solution(u, v)
    fig, ax = plt.subplots(figsize=(600 / 72.009, 269 / 72.009))
    ax.set_position([0.13, 0.14, 0.775, 0.775])
    ax.tick_params(labelsize=12)
    matlab_plot(u_display, ax=ax, linewidth=2, label="u")
    # MATLAB chebmatrix/plot promotes numeric blocks only for plotting.
    matlab_plot(cj.chebfun(v0, domain=d), ax=ax, linewidth=2, label="v")
    ax.axis([0, 1, 0, 1.05 * v0])
    ax.set_title("Solution u and radius v", fontsize=14)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u", fontsize=14)
    ax.legend(
        loc="upper right",
        fontsize=12,
        handlelength=1,
        handletextpad=0.3,
        borderpad=0.2,
        framealpha=1,
    )
    fig.set_facecolor("white")
    _savefig(fig, os.path.join(_IMG, "LaneEmden_02.png"), size=(600, 269), dpi=72.009)
    plt.close(fig)
    print(f"Polytropic range for white dwarfs: [0,{v0:1.12f})")


if __name__ == "__main__":
    run()
