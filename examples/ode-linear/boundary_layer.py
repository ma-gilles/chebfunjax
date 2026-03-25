"""Boundary layer: advection-diffusion equation.

Solves eps*u'' + u' = 0, u(0)=0, u(1)=1 for small eps.
Exact: u = (1 - exp(-x/eps)) / (1 - exp(-1/eps)).

Credit: Inspired by Chebfun ode-linear/BoundaryLayer.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import plot
from chebfunjax.operators.chebop import Chebop


def run():
    print("=" * 60)
    print("Boundary layer: eps*u'' + u' = 0")
    print("=" * 60)

    for eps in [0.1, 0.01]:
        print(f"\n--- eps = {eps} ---")
        exact = lambda x, e=eps: (1.0 - jnp.exp(-x / e)) / (1.0 - jnp.exp(-1.0 / e))
        N = Chebop(lambda x, u, e=eps: e * u.diff(2) + u.diff(), domain=(0.0, 1.0))
        N.lbc = 0.0
        N.rbc = 1.0
        u = N.solve(0.0)
        print(f"  Chebfun length: {len(u)}")

        x_test = jnp.linspace(0.0, 1.0, 500)
        exact_vals = exact(x_test)
        err = float(jnp.max(jnp.abs(u(x_test) - exact_vals)))
        print(f"  Max error vs exact: {err:.2e}")
        assert err < 1e-7, f"eps={eps} error too large: {err}"

        # Verify boundary conditions
        assert abs(float(u(jnp.array(0.0)))) < 1e-10
        assert abs(float(u(jnp.array(1.0))) - 1.0) < 1e-10

        # Find the boundary layer width (where u goes from 0.1 to 0.9)
        # Approximately at x ~ eps * log(9), but let us use roots
        f_01 = u - 0.1
        f_09 = u - 0.9
        r01 = np.array(f_01.roots())
        r09 = np.array(f_09.roots())
        if len(r01) > 0 and len(r09) > 0:
            bl_width = float(r09[0]) - float(r01[0])
            print(f"  BL width (10%-90%): {bl_width:.6f}")

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _colors = ["#4169E1", "#E04040"]
    for _eps, _col in zip([0.1, 0.01], _colors):
        _exact = lambda x, e=_eps: (1.0 - _np.exp(-x / e)) / (1.0 - _np.exp(-1.0 / e))
        _xs = _np.linspace(0.0, 1.0, 500)
        ax.plot(_xs, _exact(_xs), color=_col, linewidth=1.5,
                label=f"ε = {_eps}")
    ax.set_title("Boundary layer: ε u″ + u′ = 0", fontsize=11)
    ax.set_xlabel("x", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "boundary_layer.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
