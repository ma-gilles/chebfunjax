"""Exact T1 evaluation on independently captured native physical maps.

Provenance
----------
MATLAB source: @mapping/mapping.m, @bndfun/feval.m, @separableApprox/feval.m
Chebfun commit: 7574c77
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.separable_approx import SeparableApprox
from chebfunjax.tech.chebtech import Chebtech2

REFERENCE = json.loads((Path(__file__).parents[1] / "references" /
                        "bivariate_inverse_map_native.json").read_text())


def decode(values):
    return np.array([struct.unpack(">d", bytes.fromhex(value))[0] for value in values])


@pytest.mark.parametrize("case", ["case1", "case2", "case3", "case4"])
@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("mode", ["approx", "public", "jit"])
def test_native_t1_physical_evaluation(case, axis, mode):
    reference = REFERENCE[case]
    a,b = reference["domain"]
    one = Chebtech2.from_coeffs(jnp.array([1.]))
    t1 = Chebtech2.from_coeffs(jnp.array([0.,1.]))
    domain = (-1.,1.,a,b) if axis==1 else (a,b,-1.,1.)
    approx = SeparableApprox(cols=[t1 if axis==1 else one],
                            rows=[one if axis==1 else t1],
                            pivots=jnp.array([1.]),pivot_values=jnp.array([1.]),
                            domain=domain)
    f = Chebfun2(approx=approx)
    points = jnp.asarray(decode(reference["input_hex"]))
    x,y = (jnp.zeros_like(points),points) if axis==1 else (points,jnp.zeros_like(points))
    if mode=="approx":
        actual = approx(x,y)
    elif mode=="public":
        actual = f(x,y)
    else:
        actual = jax.jit(lambda u,v:f(u,v))(x,y)
    np.testing.assert_array_equal(np.asarray(actual),decode(reference["output_hex"]))


@pytest.mark.parametrize("case", ["case1", "case2", "case3", "case4"])
@pytest.mark.parametrize("axis", [1, 2])
def test_source_map_derivative(case, axis):
    reference = REFERENCE[case]
    a,b = reference["domain"]
    one = Chebtech2.from_coeffs(jnp.array([1.]))
    t1 = Chebtech2.from_coeffs(jnp.array([0.,1.]))
    domain = (-1.,1.,a,b) if axis==1 else (a,b,-1.,1.)
    f = Chebfun2(approx=SeparableApprox(
        cols=[t1 if axis==1 else one], rows=[one if axis==1 else t1],
        pivots=jnp.array([1.]), pivot_values=jnp.array([1.]), domain=domain))
    points = jnp.asarray(decode(reference["input_hex"]))
    evaluate = (lambda z:f(0.,z)) if axis==1 else (lambda z:f(z,0.))
    actual = jax.jit(jax.vmap(jax.grad(evaluate)))(points)
    np.testing.assert_allclose(np.asarray(actual),2/(b-a),rtol=4*np.finfo(float).eps,atol=0.)
