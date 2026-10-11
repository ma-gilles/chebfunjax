"""MATLAB expression log/heaviside regression controls.

Provenance
----------
MATLAB source: numeric log; @chebfun/heaviside.m and approx/Pushnitski.m.
Chebfun commit: 7574c77.
These controls cover eager continuation and fixed-real positive JIT;
traced negative-real continuation is explicitly outside that contract.
"""

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import _string_op
from chebfunjax.utils.matlab_expr import _source_log, matlab_expression


def test_literal_expression_log_and_heaviside():
    positive = jnp.array([0.0, 1.0, 2.0])
    v = _source_log(positive)
    assert v.dtype == jnp.float64
    assert jnp.isneginf(v[0])
    assert v[1] == 0
    mixed = jnp.array([-1.0, 0.0, 1.0])
    v = _source_log(mixed)
    assert v.dtype == jnp.complex128
    assert v[0] == 1j * jnp.pi
    assert jnp.isneginf(v[1].real)
    assert v[1].imag == 0
    assert v[2] == 0
    complexx = jnp.array([-1 + 1j, 2 - 1j])
    assert jnp.array_equal(_source_log(complexx), jnp.log(complexx))
    assert jax.jit(_source_log)(jnp.array([1.0, 2.0])).dtype == jnp.float64
    assert jnp.isnan(jax.jit(_source_log)(jnp.array([-1.0])))[0]
    step = _string_op("heaviside(x)")(mixed)
    assert jnp.array_equal(step, jnp.array([0.0, 0.5, 1.0]))
    f = cj.chebfun("log(x)", domain=[1.0, 2.0], n=17)
    assert not jnp.iscomplexobj(f.funs[0].coeffs)
    assert (
        jnp.max(jnp.abs(f(jnp.array([1.0, 1.5, 2.0])) - jnp.log(jnp.array([1.0, 1.5, 2.0]))))
        < 1e-14
    )
    f = cj.chebfun("log(x)", domain=[-2.0, -1.0], n=17)
    assert jnp.iscomplexobj(f.funs[0].coeffs)
    assert (
        jnp.max(
            jnp.abs(f(jnp.array([-2.0, -1.5, -1.0])) - jnp.log(jnp.array([-2.0, -1.5, -1.0]) + 0j))
        )
        < 1e-14
    )
    values = _string_op("-heaviside(x)./log(x)")(jnp.array([-0.1, -0.05, 0.05, 0.1]))
    assert jnp.all(jnp.isfinite(values))
    assert jnp.all(values[:2] == 0)
    assert jnp.all(values.imag == 0)


def test_expression_arithmetic_preserves_exact_real_storage():
    x = jnp.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    v = matlab_expression("-heaviside(x)./log(x)", ("x",))(x)
    assert not jnp.iscomplexobj(v)
    assert jnp.all(jnp.isfinite(v)) and jnp.all(v[:3] == 0)
    z = jnp.array([1 + 0j, 2 + 0j])
    assert jnp.iscomplexobj(matlab_expression("x", ("x",))(z))
    for expr in ["x.*1", "x./1"]:
        assert not jnp.iscomplexobj(matlab_expression(expr, ("x",))(z))
        tiny = z + 1e-30j
        result = matlab_expression(expr, ("x",))(tiny)
        assert jnp.iscomplexobj(result) and jnp.array_equal(result, tiny)
        true = jnp.array([1 + 2j, -3 + 4j])
        assert jnp.array_equal(matlab_expression(expr, ("x",))(true), true)
    f = cj.chebfun("-heaviside(x)./log(x)", domain=[-0.1, 0.1], n=1000)
    assert all(not jnp.iscomplexobj(piece.coeffs) for piece in f.funs)
    assert not jnp.iscomplexobj(f(jnp.array([-0.05, 0.05])))
    assert jnp.iscomplexobj(cj.chebfun(jnp.array([1 + 0j, 2 + 0j])).funs[0].coeffs)
