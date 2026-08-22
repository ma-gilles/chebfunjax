"""Port of MATLAB Chebfun tests/ballfun/test_coeffs3.m (Fable 5).

MATLAB's ``ballfun(F, "coeffs")`` maps to ``Ballfun.from_coeffs``.

Provenance
----------
MATLAB source : tests/ballfun/test_coeffs3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

jax.config.update("jax_enable_x64", True)


def _heavy():
    return Ballfun.from_function(
        lambda r, lam, th: -2.0 * (
            2.0 * r ** 2 * jnp.cos(lam) ** 2
            * jnp.cos(r ** 2 * jnp.cos(lam) ** 2 * jnp.sin(th) ** 2)
            * jnp.sin(th) ** 2
            + jnp.sin(r ** 2 * jnp.cos(lam) ** 2 * jnp.sin(th) ** 2)
        ) + 4.0 * jnp.cos(r ** 2 * jnp.sin(th) ** 2 * jnp.cos(lam) ** 2),
        spherical=True,
    )


class TestBallfunCoeffs3:
    def test_all_matlab_assertions(self):
        # Example 1
        f = Ballfun.from_function(lambda r, lam, th: 1.0 + 0 * r,
                                  spherical=True)
        F = np.array(f.coeffs3(3, 4, 5))
        exact = np.zeros((3, 4, 5), dtype=complex)
        exact[0, 2, 2] = 1.0
        assert np.max(np.abs(F - exact)) < 1e-14  # pass(1)

        # Example 2
        exact_f = _heavy()
        F = f2 = None
        F = exact_f.coeffs3(50, 51, 52)
        f2 = Ballfun.from_coeffs(F)
        assert f2.isequal(exact_f)  # pass(2)
        assert tuple(np.array(F).shape) == (50, 51, 52)

        # Example 3: no-argument coeffs3 returns the raw tensor
        F = exact_f.coeffs3()
        f3 = Ballfun.from_coeffs(F)
        assert f3.isequal(exact_f)  # pass(3)

        # Example 4: single size argument
        F = exact_f.coeffs3(50)
        f4 = Ballfun.from_coeffs(F)
        assert f4.isequal(exact_f)  # pass(4)
        assert tuple(np.array(F).shape) == (50, 50, 50)

        # Example 5: two size arguments
        F = exact_f.coeffs3(50, 51)
        f5 = Ballfun.from_coeffs(F)
        assert f5.isequal(exact_f)  # pass(5)
        assert tuple(np.array(F).shape) == (50, 51, 51)
