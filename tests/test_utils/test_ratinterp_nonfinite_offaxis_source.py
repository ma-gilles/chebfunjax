"""Bounded off-axis source primitives and independent public complex fit.

Provenance
----------
MATLAB source : ratinterp.m/ratbary, Chebfun commit: 7574c77
Actual R2025b captures; axis/zero factors intentionally outside new primitive.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._ratbary import _source_nonfinite_offaxis, _source_type2_rational_handle
from chebfunjax.utils.ratapprox import ratinterp

FIX = Path(__file__).parents[1]/"fixtures"/"ratinterp_nonfinite_offaxis_2025b"

def decode(r):
    def words(k):
        return np.array([struct.unpack(">d", bytes.fromhex(w))[0] for w in r[k]])
    return words("real_hex"), words("imag_hex")

def array(r):
    re, im = decode(r)
    return jax.lax.complex(jnp.asarray(re), jnp.asarray(im))

def check(actual, expected):
    re, im = decode(expected)
    actual = np.asarray(actual).reshape(-1)
    np.testing.assert_array_equal(actual.real, re)
    np.testing.assert_array_equal(actual.imag, im)

def test_captured_nonfinite_offaxis_primitives():
    records = json.loads((FIX/"complex_primitives.json").read_text())["records"]
    def multiply(u, v):
        return _source_nonfinite_offaxis(u, v)

    def divide(u, v):
        return _source_nonfinite_offaxis(u, v, divide=True)

    operations = ((multiply, jax.jit(multiply), "times"),
                  (divide, jax.jit(divide), "rdivide"))
    used = 0
    for r in records:
        re, im = decode(r["v"])
        if not (np.all(np.isfinite(re)) and np.all(np.isfinite(im)) and np.all(re != 0) and np.all(im != 0)):
            continue  # Declared scope selection, not runtime failure skipping.
        used += 1
        for eager, compiled, field in operations:
            for run in (eager, compiled):
                check(run(array(r["u"]), array(r["v"])), r[field])
    assert used == 18

def test_actual_public_complex_source_output():
    source = json.loads((FIX/"type2_public.json").read_text())
    z, w = .2+.3j, 1.5-.4j
    r, a, b, mu, nu, poles, _ = ratinterp(lambda x: (1+2j)/((x-z)*(x-w)), 2, 2, xi="type2")
    assert (mu, nu) == (0, 2)
    np.testing.assert_allclose(np.sort_complex(poles), np.sort_complex([z, w]), atol=1e-10, rtol=0)
    x = jnp.array([.1+.2j, -.5+.1j])
    for run in (r, jax.jit(r)):
        check(run(x), source["r_at_query"])


def test_captured_scalar_stage_operations():
    # Captured operands here isolate arithmetic; never used by public fitting.
    stage = json.loads((FIX/"all_stages.json").read_text())["stage"]
    for i in (1, 2):
        for compiled in (False, True):
            def multiply(u, v):
                return _source_nonfinite_offaxis(u, v)

            def divide(u, v):
                return _source_nonfinite_offaxis(u, v, divide=True)
            if compiled:
                multiply, divide = jax.jit(multiply), jax.jit(divide)
            check(multiply(array(stage["pxw"]), array(stage[f"dxpinv{i}"])), stage[f"numeratorDot{i}"])
            check(divide(array(stage[f"numeratorDot{i}"]), array(stage[f"denominatorDot{i}"])), stage[f"afterDivide{i}"])
            check(multiply(array(stage["yAfterDivide"]), array(stage["lp"])), stage["afterLp"])
            check(divide(array(stage["afterLp"]), array(stage["lq"])), stage["separateFinalDivide"])


def test_actual_public_high_degree_source_output():
    source = json.loads((FIX / "default_high_degree_public.json").read_text())
    r, _, _, mu, nu, _, _ = ratinterp(lambda x: 1.0 / (x - 0.3), 8, 4)
    assert (mu, nu) == (0, 1)
    x = jnp.asarray(np.linspace(-0.9, 0.2, 15))
    for run in (r, jax.jit(r)):
        check(run(x), source["r_at_query"])


@pytest.mark.parametrize("compiled", [False, True])
def test_finite_complex_public_query_derivatives(compiled):
    a0, a1 = 1 + .5j, .25 - .125j
    b0, b1, b2 = 2 + .25j, .2 - .1j, .125 + .0625j
    r = _source_type2_rational_handle(
        jnp.array([a0, a1]), jnp.array([b0, b1, b2]), (-1., 1.))
    x, direction, cotangent = jnp.array(.17 + .23j), jnp.array(.3 - .4j), jnp.array(.7 + .2j)
    a = a0 + a1*x
    b = b0 + b1*x + b2*(2*x*x - 1)
    slope = (a1*b - a*(b1 + 4*b2*x)) / (b*b)

    def forward(z, dz):
        return jax.jvp(r, (z,), (dz,))

    def reverse(z, dz):
        value, pullback = jax.vjp(r, z)
        return value, pullback(dz)[0]

    if compiled:
        forward, reverse = jax.jit(forward), jax.jit(reverse)
    value, tangent = forward(x, direction)
    reverse_value, adjoint = reverse(x, cotangent)
    np.testing.assert_allclose(value, a/b, atol=1e-13, rtol=0)
    np.testing.assert_allclose(reverse_value, a/b, atol=1e-13, rtol=0)
    np.testing.assert_allclose(tangent, slope*direction, atol=1e-13, rtol=0)
    np.testing.assert_allclose(adjoint, slope*cotangent, atol=1e-13, rtol=0)
