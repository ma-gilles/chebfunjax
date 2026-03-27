"""Roots of Bessel function J_0.

Demonstrates finding the zeros of J_0(x) on [0, 100] using
Chebfun's rootfinding and comparing with scipy.

Credit: Inspired by Chebfun examples roots/BesselRoots.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import scipy.special as sp
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.plotting import chebfun_style
chebfun_style()

from chebfunjax.plotting import plot

def run():
    print("=" * 60)
    print("Roots of Bessel function J_0 on [0, 100]")
    print("=" * 60)

    from scipy.special import j0 as besselj0, jn_zeros

    # MATLAB: J0 = chebfun(@(x) besselj(0,x), [0 100])
    J0 = cj.chebfun(
        lambda x: jnp.array(besselj0(np.array(x))),
        domain=(0.0, 100.0),
    )
    print(f"\nJ_0(x) on [0, 100]:")
    print(f"  Chebfun length: {len(J0)}")

    # MATLAB: r = roots(J0)
    r = J0.roots()
    r_arr = np.sort(np.array(r))
    print(f"  Number of roots found: {len(r_arr)}")

    # Reference roots from scipy
    n_roots = min(len(r_arr), 30)
    ref_roots = jn_zeros(0, n_roots)

    print(f"\n  First 10 roots of J_0 (vs scipy reference):")
    print(f"  {'k':>4}  {'Chebfun root':>20}  {'scipy root':>20}  {'Error':>12}")
    for k in range(min(10, n_roots)):
        err = abs(r_arr[k] - ref_roots[k])
        print(f"  {k+1:>4}  {r_arr[k]:>20.14f}  {ref_roots[k]:>20.14f}  {err:.2e}")

    # Verify all roots match scipy to high accuracy
    max_err = np.max(np.abs(r_arr[:n_roots] - ref_roots[:n_roots]))
    print(f"\n  Max error vs scipy: {max_err:.2e}")
    assert max_err < 1e-9, f"Root error too large: {max_err}"

    # Verify that f(root) ~ 0
    max_val_at_roots = float(np.max(np.abs(np.array([
        float(J0(jnp.array(ri))) for ri in r_arr[:n_roots]
    ]))))
    print(f"  Max |J_0(root)| = {max_val_at_roots:.2e}  (should be ~0)")
    assert max_val_at_roots < 1e-10

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(J0, title="J₀(x) and its zeros on [0, 100]")
    ax.axhline(0, color="k", linewidth=0.5)
    _r_arr = r_arr[:30]  # first 30 roots
    import numpy as _np
    ax.plot(_r_arr, _np.zeros_like(_r_arr), "o", color="#E04040",
            markersize=3, label="roots")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "bessel_roots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True

if __name__ == "__main__":
    run()
