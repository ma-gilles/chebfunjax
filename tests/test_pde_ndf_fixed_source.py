"""pdeSolve fixed-N prolong/output controls and ordinary-ODE compatibility."""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun1d.pde15s import pde15s


@pytest.mark.parametrize("coefficients,n", [([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 4), ([1.0, 2.0], 8)])
def test_fixed_n_prolongs_initial_coefficients_and_retains_output_size(coefficients, n):
    original = Chebfun.from_coeffs(jnp.asarray(coefficients))
    result = pde15s(lambda u: 0 * u, [0.0, 0.1], original, n=n)
    expected = jnp.pad(jnp.asarray(coefficients[:n]), (0, max(0, n - len(coefficients))))
    assert len(result) == 2
    assert all(len(f) == n for f in result)
    assert bool(jnp.array_equal(result[0].funs[0].tech.coeffs, expected))
    assert len(original) == len(coefficients)


def test_no_bc_nonautonomous_manufactured_solution():
    original = chebfun(lambda x: x)
    result = pde15s(lambda t, x, u: 0 * u + 1 + t, [0.0, 0.2], original, n=16, rtol=1e-7, atol=1e-7)
    x = jnp.linspace(-1.0, 1.0, 21)
    assert float(jnp.max(jnp.abs(result[-1](x) - (x + 0.22)))) < 1e-6
