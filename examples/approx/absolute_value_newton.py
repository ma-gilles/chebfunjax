"""Approximation of |x| via Newton iteration.

Demonstrates Newton's method applied to the function equation
r^2 = x^2, i.e., r := (r^2 + x^2) / (2*r), converging to |x|.

Credit: Inspired by Chebfun approx/AbsNewton.m (Nick Trefethen).
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

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
    print("Newton iteration converging to |x|")
    print("=" * 60)

    dom = (-1.0, 1.0)
    x = cj.chebfun(lambda t: t, domain=dom)

    # Newton iteration: r_{n+1} = (r_n^2 + x^2) / (2 * r_n)
    # Start from r_0 = 1 (constant chebfun)
    r = cj.chebfun(lambda t: jnp.ones_like(t), domain=dom)

    print(f"\nNewton iteration for |x| starting from r_0 = 1:")
    print(f"  {'Step':>4}  {'||r - |x|||_inf':>20}  {'length':>8}")

    x_test = jnp.linspace(-1.0, 1.0, 500)
    abs_x = jnp.abs(x_test)

    n_steps = 25
    for k in range(n_steps):
        r_new = (r**2 + x**2) / (2.0 * r)
        err = float(jnp.max(jnp.abs(r_new(x_test) - abs_x)))
        if k < 5 or k % 5 == 0:
            print(f"  {k+1:>4}  {err:>20.4e}  {len(r_new):>8}")
        r = r_new
        if err < 1e-13:
            print(f"  Converged at step {k+1}!")
            break

    final_err = float(jnp.max(jnp.abs(r(x_test) - abs_x)))
    print(f"\nFinal ||r - |x|||_inf = {final_err:.2e}")

    # Verify the result matches |x|
    f_abs_ref = cj.chebfun(lambda t: jnp.abs(t), domain=(-1.0, 0.0, 1.0))
    x_interior = jnp.linspace(-0.99, 0.99, 200)
    err_vs_abs = float(jnp.max(jnp.abs(r(x_interior) - f_abs_ref(x_interior))))
    print(f"||r - |x|_chebfun||_inf at interior pts: {err_vs_abs:.2e}")
    assert err_vs_abs < 1e-10, f"Newton iteration did not converge: {err_vs_abs}"

    # --- Plots -------------------------------------------------------
    _here = os.path.dirname(os.path.abspath(__file__))
    fig, ax = plot(r, title="Newton iteration converging to |x|", label="r (converged)")
    _abs_cheb = cj.chebfun(lambda t: jnp.abs(t), domain=(-1.0, 0.0, 1.0))
    import numpy as _np
    _xs = _np.linspace(-1.0, 1.0, 600)
    ax.plot(_xs, _np.abs(_xs), "--", color="#E04040", linewidth=1.2, label="|x|")
    ax.legend(fontsize=9)
    fig.savefig(os.path.join(_here, "absolute_value_newton.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
