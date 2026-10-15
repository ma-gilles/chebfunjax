"""Scalar matrix division against independent pinned MATLAB captures.

Provenance
----------
MATLAB source: @chebfun2v/mrdivide.m, @separableApprox/mrdivide.m
Chebfun commit: 7574c77
"""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref

REFERENCE = json.loads((Path(__file__).parents[2] / "references" /
                        "chebfun2v_scalar_division_native.json").read_text())


@pytest.mark.parametrize("value,key", [(2., "mrdivide_real"),
                                       (2.+1j, "mrdivide_complex")])
@pytest.mark.parametrize("jax_scalar", [False, True])
def test_scalar_division_native_coefficients_and_values(value, key, jax_scalar):
    field = Chebfun2v.from_functions(lambda x, y: 2+jnp.cos(x),
                                    lambda x, y: 3+jnp.sin(y))
    divisor = jnp.asarray(value) if jax_scalar else value
    result = field / divisor
    native = REFERENCE["cases"][key]
    tol = 1e3*ChebfunPref().cheb2Prefs.chebfun2eps
    x = jnp.array([-.75, .2, .8])
    y = jnp.array([.1, -.4, .6])
    for component, values, coefficients in zip(
            result.components, native["values"], native["coefficients"]):
        f = Chebfun2(approx=component)
        expected = jnp.asarray(values["real"])+1j*jnp.asarray(values["imag"])
        assert float(jnp.linalg.norm(f(x,y)-expected)) < tol
        expected = (jnp.asarray(coefficients["real"])
                    +1j*jnp.asarray(coefficients["imag"]))
        actual = f.coeffs2().reshape(-1)
        assert actual.shape == expected.shape
        assert float(jnp.linalg.norm(actual-expected)) < tol
