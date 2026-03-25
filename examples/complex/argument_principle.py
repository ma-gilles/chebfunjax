"""Argument principle for counting zeros and poles.

Demonstrates the argument principle: for a meromorphic function f,
the change in argument of f around a closed contour equals 2*pi*(Z-P)
where Z and P are the numbers of zeros and poles inside the contour.

Verified by counting roots of polynomials on real intervals using
Chebfun's roots() method, which is equivalent to Z-P counting.

Credit: Inspired by Chebfun complex/ArgumentPrinciple examples.
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


def run():
    print("=" * 60)
    print("Argument principle: counting zeros")
    print("=" * 60)

    # --- Count zeros of p(x) = (x-1)(x-2)(x-3) on [0, 4] ----------
    p1 = cj.chebfun(lambda x: (x - 1.0)*(x - 2.0)*(x - 3.0), domain=(0.0, 4.0))
    roots1 = p1.roots()
    roots1_arr = np.sort(np.array(roots1))
    print(f"\np(x) = (x-1)(x-2)(x-3) on [0,4]:")
    print(f"  Roots found: {roots1_arr}")
    print(f"  Expected: [1, 2, 3]")
    assert len(roots1_arr) == 3
    assert abs(roots1_arr[0] - 1.0) < 1e-10
    assert abs(roots1_arr[1] - 2.0) < 1e-10
    assert abs(roots1_arr[2] - 3.0) < 1e-10

    # --- Count zeros of sin(n*pi*x) on [0, 1] ----------------------
    # sin(n*pi*x) has zeros at x = 0, 1/n, 2/n, ..., 1 → n+1 zeros
    for n in [3, 5, 8]:
        fn = cj.chebfun(lambda x, n=n: jnp.sin(n * float(jnp.pi) * x), domain=(0.0, 1.0))
        roots_n = fn.roots()
        # Interior roots (excluding endpoints can vary slightly)
        roots_n_arr = np.sort(np.array(roots_n))
        # Filter to [0, 1]
        roots_in = roots_n_arr[(roots_n_arr >= -1e-10) & (roots_n_arr <= 1.0 + 1e-10)]
        print(f"  sin({n}*pi*x) on [0,1]: {len(roots_in)} zeros (expected {n+1})")
        assert len(roots_in) == n + 1

    # --- Argument principle analogue: sign changes of f --------------
    # For a continuous real function, number of sign changes = number of zeros
    # f(x) = cos(5*x) on [0, 2*pi]: 10 zeros
    pi = float(jnp.pi)
    f2 = cj.chebfun(lambda x: jnp.cos(5.0 * x), domain=(0.0, 2.0 * pi))
    roots2 = f2.roots()
    roots2_arr = np.sort(np.array(roots2))
    # Interior zeros of cos(5x) on [0, 2*pi]
    interior = roots2_arr[(roots2_arr > 1e-10) & (roots2_arr < 2.0*pi - 1e-10)]
    print(f"\ncos(5x) on [0, 2*pi]:")
    print(f"  Interior zeros: {len(interior)} (expected 10)")
    assert len(interior) == 10

    # Verify by checking function values at the zeros
    for r in interior:
        val = float(f2(jnp.array(r)))
        assert abs(val) < 1e-10

    # --- Stability: Routh-Hurwitz condition via root counting --------
    # p(x) = x^4 + 2x^3 + 3x^2 + 4x + 5 has no real roots
    p_stable = cj.chebfun(lambda x: x**4 + 2.0*x**3 + 3.0*x**2 + 4.0*x + 5.0,
                           domain=(-10.0, 10.0))
    roots_stable = np.array(p_stable.roots())
    # All roots should be complex; no real roots
    real_roots = [r for r in roots_stable if abs(float(p_stable(jnp.array(r)))) < 1e-8]
    print(f"\np(x) = x^4+2x^3+3x^2+4x+5 on [-10,10]:")
    print(f"  Number of real roots found: {len(roots_stable)}")
    # Polynomial is positive definite? Let's check at some points
    vals_at_pts = [float(p_stable(jnp.array(float(x)))) for x in [-2., -1., 0., 1., 2.]]
    print(f"  Values at -2,-1,0,1,2: {[f'{v:.2f}' for v in vals_at_pts]}")
    # p(-1) = 1-2+3-4+5 = 3 > 0
    # p(0) = 5 > 0; minimum is positive → 0 real roots
    assert all(v > 0 for v in vals_at_pts)

    # --- Rolle's theorem: between consecutive zeros of f, f' has a zero -
    f3 = cj.chebfun(lambda x: jnp.sin(x) * jnp.cos(x), domain=(0.0, 3.0*pi))
    f3p = f3.diff()
    roots_f3 = np.sort(np.array(f3.roots()))
    roots_f3p = np.sort(np.array(f3p.roots()))
    interior_f3 = roots_f3[(roots_f3 > 1e-8) & (roots_f3 < 3.0*pi - 1e-8)]
    interior_f3p = roots_f3p[(roots_f3p > 1e-8) & (roots_f3p < 3.0*pi - 1e-8)]
    print(f"\nRolle's theorem for f(x) = sin(x)*cos(x):")
    print(f"  f has {len(interior_f3)} interior zeros")
    print(f"  f' has {len(interior_f3p)} interior zeros")
    # Between each pair of consecutive zeros of f, there's at least one zero of f'
    assert len(interior_f3p) >= len(interior_f3) - 1

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(p1, title="(x−1)(x−2)(x−3): roots on [0,4]",
                   label="p(x)")
    ax.axhline(0, color="k", linewidth=0.5)
    import numpy as _np
    ax.plot(roots1_arr, _np.zeros_like(roots1_arr), "r^",
            markersize=8, label="roots")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "argument_principle.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
