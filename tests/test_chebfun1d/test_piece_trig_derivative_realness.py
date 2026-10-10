"""Physical periodic differentiation preserves native value-realness metadata.

Provenance
----------
MATLAB source : @trigtech/populate.m (isreal(rndVal(k))), @bndfun/diff.m,
                @trigtech/diff.m (continuous dimension preserves isReal).
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Complex storage with an exactly zero imaginary probe is native-real; genuine
imaginary input remains complex. Storage dtype alone is not this contract.
"""
import jax.numpy as jnp
import pytest

import chebfunjax as cj


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("input_kind", ["real", "complex_zero", "complex_nonzero"])
def test_periodic_piece_derivative_keeps_source_realness(order, input_kind):
    def op(x):
        value = jnp.sin(x)
        if input_kind == "complex_zero":
            return value.astype(jnp.complex128)
        if input_kind == "complex_nonzero":
            return value + 1j*jnp.cos(x)
        return value
    f = cj.chebfun(op, domain=(0., 2*jnp.pi), trig=True)
    derivative = f.diff(order)
    xx = jnp.array([0., .2, 1.3, 2*jnp.pi])
    values = derivative(xx)
    source_real = input_kind != "complex_nonzero"
    assert f.funs[0].tech.is_real == source_real
    assert derivative.funs[0].tech.is_real == source_real
    assert jnp.iscomplexobj(values) == (not source_real)
    expected = jnp.sin(xx + order*jnp.pi/2)
    if input_kind == "complex_nonzero":
        expected = expected + 1j*jnp.cos(xx + order*jnp.pi/2)
    assert float(jnp.max(jnp.abs(values-expected))) < 5e-14
