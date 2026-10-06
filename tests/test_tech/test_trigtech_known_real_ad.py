"""Controls for the explicitly real JAX objective adapter.

MATLAB has no JAX differentiation API. These independent Fourier identities
qualify the documented adapter needed when complex coefficients are traced.
They do not assert value-dependent static realness under tracing.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('explicit_real', [True, False], ids=['known-real-flag', 'real-objective'])
def test_complex_coefficient_gradient_matches_independent_fourier_phase(explicit_real):
    x = jnp.asarray(0.3, dtype=jnp.float64)
    coeffs = jnp.asarray([0.5, 0.0, 0.5], dtype=jnp.complex128)

    def objective(c):
        if explicit_real:
            return Trigtech.from_coeffs(c, is_real=True)(x)
        return jnp.real(Trigtech.from_coeffs(c)(x))

    result = jax.jit(jax.grad(objective))(coeffs)
    # The known-real Horner branch uses only c[-1], c[0] and their
    # real/imaginary parts: 2*Re(c[-1]*exp(-i*pi*x))+Re(c[0]). Its
    # redundant positive coefficient is ignored, so the gradient is
    # [2*exp(-i*pi*x),1,0]. For the unconstrained complex Fourier branch,
    # Re(sum(c[k]*exp(i*pi*k*x))) has gradient exp(i*pi*k*x).
    if explicit_real:
        expected = jnp.asarray([2*jnp.exp(-1j*jnp.pi*x), 1.0, 0.0])
    else:
        expected = jnp.exp(1j * jnp.pi * jnp.asarray([-1.0, 0.0, 1.0]) * x)
    assert result.shape == coeffs.shape
    assert bool(jnp.all(jnp.abs(result-expected) <= 100*jnp.finfo(jnp.float64).eps))
