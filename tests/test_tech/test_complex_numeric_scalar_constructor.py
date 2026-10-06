"""Controls for phase-preserving complex scalar Chebfun construction.

The trig constant case is MATLAB tests/trigtech/test_feval.m pass 14. The
nontrig scalar case checks the same public numeric-constant constructor on a
finite nondefault domain. These are focused API controls, not a claim of broad
MATLAB constructor parity.

Provenance
----------
MATLAB source : @chebfun/chebfun.m, @chebfun/constructor.m,
                tests/trigtech/test_feval.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.tech.trigtech import Trigtech

CONSTANT = 1.0 + 2.0j
DOMAINS = [(False, (-3.0, 5.0)), (True, (0.0, 2.0 * np.pi))]


def _points(shape, domain):
    a, b = domain
    if shape == "scalar":
        unit = jnp.asarray(0.25)
    elif shape == "vector":
        unit = jnp.asarray([0.1, 0.25, 0.75])
    else:
        unit = jnp.asarray([[0.1, 0.25], [0.5, 0.75]])
    return a + (b - a) * unit


@pytest.mark.parametrize("trig,domain", DOMAINS, ids=["chebfun", "trig"])
@pytest.mark.parametrize("shape", ["scalar", "vector", "matrix"])
def test_complex_scalar_preserves_phase_domain_and_input_shape(trig, domain, shape):
    f = chebfun(CONSTANT, domain=domain, trig=trig)
    x = _points(shape, domain)
    actual = jnp.asarray(f(x))
    expected = jnp.full(x.shape, CONSTANT, dtype=jnp.complex128)

    assert actual.shape == x.shape
    assert jnp.issubdtype(actual.dtype, jnp.complexfloating)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert not f.isreal()
    np.testing.assert_array_equal(
        np.asarray(f.domain.breakpoints), np.asarray(domain, dtype=np.float64)
    )
    if trig:
        assert isinstance(f.funs[0].tech, Trigtech)
        np.testing.assert_array_equal(
            np.asarray(f.funs[0].tech.coeffs), np.asarray([CONSTANT])
        )


def test_complex_trig_scalar_uses_default_domain_and_zero_mode():
    f = chebfun(CONSTANT, trig=True)
    np.testing.assert_array_equal(
        np.asarray(f.domain.breakpoints), np.asarray([-1.0, 1.0])
    )
    tech = f.funs[0].tech
    assert isinstance(tech, Trigtech)
    np.testing.assert_array_equal(np.asarray(tech.coeffs), np.asarray([CONSTANT]))
    assert not f.isreal()


def test_real_scalar_trig_flag_selects_trigtech():
    f = chebfun(2.5, trig=True)
    assert isinstance(f.funs[0].tech, Trigtech)
    assert f.funs[0].tech.is_real
    np.testing.assert_array_equal(
        np.asarray(f.funs[0].tech.coeffs), np.asarray([2.5 + 0.0j])
    )
    np.testing.assert_array_equal(
        np.asarray(f.domain.breakpoints), np.asarray([-1.0, 1.0])
    )


@pytest.mark.parametrize(
    "constant", [CONSTANT, jnp.asarray(CONSTANT)],
    ids=["python-complex", "jax-scalar-complex"],
)
def test_nontrig_complex_scalar_inputs_preserve_phase(constant):
    f = chebfun(constant, domain=(-2.0, 3.0))
    points = jnp.asarray([-1.0, 0.0, 2.0])
    actual = jnp.asarray(f(points))
    expected = jnp.full(points.shape, CONSTANT, dtype=jnp.complex128)
    assert actual.shape == points.shape
    assert jnp.issubdtype(actual.dtype, jnp.complexfloating)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert not f.isreal()


def test_jax_complex_scalar_trig_input_uses_complex_zero_mode():
    f = chebfun(jnp.asarray(CONSTANT), trig=True)
    assert isinstance(f.funs[0].tech, Trigtech)
    np.testing.assert_array_equal(
        np.asarray(f.funs[0].tech.coeffs), np.asarray([CONSTANT])
    )
    assert not f.isreal()
