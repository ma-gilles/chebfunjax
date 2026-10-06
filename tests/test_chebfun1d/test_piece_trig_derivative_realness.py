"""Physical periodic differentiation preserves source real/complex state.

Provenance
----------
MATLAB source : @bndfun/diff.m, @trigtech/diff.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("complex_input", [False, True])
def test_periodic_piece_derivative_keeps_source_realness(order, complex_input):
    def op(x):
        value = jnp.sin(x)
        return value.astype(jnp.complex128) if complex_input else value
    f = cj.chebfun(op, domain=(0., 2*jnp.pi), trig=True)
    derivative = f.diff(order)
    xx = jnp.array([0., .2, 1.3, 2*jnp.pi])
    values = derivative(xx)
    assert jnp.iscomplexobj(values) == complex_input
    expected = jnp.sin(xx + order*jnp.pi/2)
    assert float(jnp.max(jnp.abs(values-expected))) < 5e-14
