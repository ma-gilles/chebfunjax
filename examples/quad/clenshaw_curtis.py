"""Clenshaw-Curtis quadrature via Chebfun sum.

Demonstrates that Chebfun's sum() method computes Clenshaw-Curtis
quadrature, which is spectrally accurate for smooth integrands.

Credit: Inspired by Chebfun examples quad/ClenshawCurtis.m.
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
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.plotting import plot

def run():
    print("=" * 60)
    print("Clenshaw-Curtis quadrature via Chebfun sum")
    print("=" * 60)

    # --- Classic integrals ------------------------------------------
    integrals = [
        # (name, f, domain, exact)
        ("exp(x) on [-1,1]",
         lambda x: jnp.exp(x), (-1.0, 1.0),
         float(jnp.exp(jnp.array(1.0)) - jnp.exp(jnp.array(-1.0)))),
        ("sin(x) on [0, pi]",
         lambda x: jnp.sin(x), (0.0, float(jnp.pi)),
         2.0),
        ("x^2 on [0, 1]",
         lambda x: x**2, (0.0, 1.0),
         1.0/3.0),
        ("1/(1+x^2) on [-1,1]",
         lambda x: 1.0/(1.0+x**2), (-1.0, 1.0),
         float(jnp.pi/2.0)),
        ("exp(-x^2) on [-5,5]",
         lambda x: jnp.exp(-x**2), (-5.0, 5.0),
         float(jnp.sqrt(jnp.pi))),
    ]

    print(f"\n{'Integral':40s} {'Computed':18} {'Exact':18} {'Error':10}")
    print("-" * 90)
    for name, f_func, (a, b), exact in integrals:
        dom = (a, b)
        f = cj.chebfun(f_func, domain=dom)
        computed = float(f.sum())
        err = abs(computed - exact)
        print(f"{name:40s} {computed:18.12f} {exact:18.12f} {err:.2e}")

    # Check each to reasonable tolerance
    for name, f_func, (a, b), exact in integrals[:4]:
        dom = (a, b)
        f = cj.chebfun(f_func, domain=dom)
        computed = float(f.sum())
        assert abs(computed - exact) < 1e-12, \
            f"Error too large for {name}: {abs(computed - exact):.2e}"

    # The Gaussian integral is approximate (domain is [-5,5], not [-inf,inf])
    dom_gauss = (-5.0, 5.0)
    f_gauss = cj.chebfun(lambda x: jnp.exp(-x**2), domain=dom_gauss)
    gauss_int = float(f_gauss.sum())
    assert abs(gauss_int - float(jnp.sqrt(jnp.pi))) < 1e-10

    # --- Chebfun lengths for different functions ---------------------
    print("\nLength (polynomial degree) of chebfun for different functions:")
    fns = [
        ("exp(x)",     lambda x: jnp.exp(x),      (-1.0, 1.0)),
        ("cos(10*x)",  lambda x: jnp.cos(10.0*x), (-1.0, 1.0)),
        ("cos(100*x)", lambda x: jnp.cos(100.0*x),(-1.0, 1.0)),
        ("1/(1+x^2)",  lambda x: 1.0/(1.0+x**2),  (-1.0, 1.0)),
    ]
    for name, f_func, (a, b) in fns:
        dom = (a, b)
        f = cj.chebfun(f_func, domain=dom)
        print(f"  {name:20s}: length = {len(f):5d}")

    # --- Spike integral ----------------------------------------------
    dom5 = (-1.0, 1.0)
    f_spike = cj.chebfun(lambda x: 1.0 / (1.0 + (100.0 * x)**2), domain=dom5)
    spike_int = float(f_spike.sum())
    exact_spike = (2.0 / 100.0) * float(jnp.arctan(jnp.array(100.0)))
    print(f"\nSpike integral 1/(1+(100x)^2) over [-1,1]:")
    print(f"  Computed: {spike_int:.12f}")
    print(f"  Exact:    {exact_spike:.12f}")
    assert abs(spike_int - exact_spike) < 1e-11

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    import matplotlib.pyplot as _plt
    import numpy as _np
    fig, ax = _plt.subplots()
    _funcs = [
        ("exp(x)",   lambda x: jnp.exp(x),       (-1.0, 1.0)),
        ("sin(x)",   lambda x: jnp.sin(x),        (0.0, float(jnp.pi))),
        ("exp(-x²)", lambda x: jnp.exp(-x**2),   (-5.0, 5.0)),
    ]
    _colors = ["#4169E1", "#E04040", "#228B22"]
    for (_name, _fn, _dom), _col in zip(_funcs, _colors):
        _f = cj.chebfun(_fn, domain=_dom)
        _xs = _np.linspace(float(_dom[0]), float(_dom[1]), 400)
        ax.plot(_xs, _np.array(_f(jnp.array(_xs))), color=_col,
                linewidth=1.5, label=_name)
    ax.set_title("Clenshaw-Curtis integrands", fontsize=11)
    ax.legend(fontsize=9)
    fig.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(os.path.join(_here, "clenshaw_curtis.png"),
                dpi=150, bbox_inches="tight")
    _plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
