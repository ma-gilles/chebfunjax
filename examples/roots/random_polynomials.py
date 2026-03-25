"""Roots of random Chebyshev polynomials.

Constructs random polynomials from Chebyshev coefficients and
finds their real roots, verifying that roots are genuine zeros.

Credit: Inspired by Chebfun examples roots/RandomPolynomials.m.
Original MATLAB Chebfun: Copyright 2017 by The University of Oxford and
The Chebfun Developers. See https://www.chebfun.org/ for Chebfun information.
"""

import jax.numpy as jnp
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import chebfunjax as cj
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.chebfun1d.chebfun import _Piece, Chebfun
from chebfunjax.domain import Domain


def run():
    print("=" * 60)
    print("Roots of random Chebyshev polynomials")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # --- Count real roots of random Chebyshev expansions ------------
    print("\nDegree  |  # real roots in (-1,1)")
    print("-" * 35)
    for n in [10, 20, 50]:
        # Random Chebyshev coefficients (decaying by 0.9^k for well-posedness)
        decay = 0.9 ** np.arange(n + 1)
        coeffs = rng.standard_normal(n + 1) * decay
        coeffs_jnp = jnp.array(coeffs)
        tech = Chebtech2.from_coeffs(coeffs_jnp)
        piece = _Piece(tech=tech, interval=(-1.0, 1.0))
        f = Chebfun(funs=[piece], domain=Domain((-1.0, 1.0)))
        roots = f.roots()
        r_arr = np.array(roots)
        interior = r_arr[(r_arr > -1.0 + 1e-10) & (r_arr < 1.0 - 1e-10)]
        n_roots = len(interior)
        print(f"  n={n:3d}: {n_roots:3d} real roots")
        # Sanity: should have <= n real roots
        assert n_roots <= n, f"Too many roots: {n_roots} > {n}"

    # --- Verify roots are genuine zeros ------------------------------
    rng2 = np.random.default_rng(123)
    n = 15
    decay = 0.85 ** np.arange(n + 1)
    coeffs = rng2.standard_normal(n + 1) * decay
    coeffs_jnp = jnp.array(coeffs)
    tech = Chebtech2.from_coeffs(coeffs_jnp)
    piece = _Piece(tech=tech, interval=(-1.0, 1.0))
    f_test = Chebfun(funs=[piece], domain=Domain((-1.0, 1.0)))
    roots = f_test.roots()
    r_arr = np.array(roots)
    interior = r_arr[(r_arr > -1.0 + 1e-10) & (r_arr < 1.0 - 1e-10)]

    print(f"\nVerifying {len(interior)} roots of random n={n} polynomial:")
    max_val_at_roots = 0.0
    for ri in interior:
        val = abs(float(f_test(jnp.array(ri))))
        max_val_at_roots = max(max_val_at_roots, val)
    print(f"  max|f(root)| = {max_val_at_roots:.2e}  (should be ~0)")
    vscale = float(jnp.max(jnp.abs(coeffs_jnp)))
    assert max_val_at_roots < 1e-8 * vscale, \
        f"Roots not genuine zeros: max|f(root)| = {max_val_at_roots}"

    # --- Specific known polynomial: T_2(x) - 0.5 = 2x^2 - 1.5 ------
    # Roots: x = +-sqrt(3/4)
    coeffs2 = jnp.array([-0.5, 0.0, 1.0])  # Chebyshev coeffs of T_2 - 0.5
    tech2 = Chebtech2.from_coeffs(coeffs2)
    piece2 = _Piece(tech=tech2, interval=(-1.0, 1.0))
    f2 = Chebfun(funs=[piece2], domain=Domain((-1.0, 1.0)))
    roots2 = f2.roots()
    r2 = np.sort(np.array(roots2))
    exact2 = np.array([-np.sqrt(3.0/4.0), np.sqrt(3.0/4.0)])
    print(f"\nRoots of T_2(x) - 0.5 = 2x^2 - 1.5:")
    print(f"  Computed: {r2}")
    print(f"  Exact: +-sqrt(3/4) = {exact2}")
    assert len(r2) == 2
    assert abs(r2[0] - exact2[0]) < 1e-12
    assert abs(r2[1] - exact2[1]) < 1e-12

    print("\nAll assertions passed.")
    return True


if __name__ == "__main__":
    run()
