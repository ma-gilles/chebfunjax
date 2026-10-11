"""Original six predicates from tests/chebfun/test_getValuesAtBreakpoints.m.

Provenance
----------
MATLAB source: tests/chebfun/test_getValuesAtBreakpoints.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = jnp.finfo(jnp.float64).eps

def test_source_1():
    f = cj.chebfun(lambda x: x, domain=[-1, 0, .5, 1])
    assert bool(jnp.all(jnp.abs(cj.getValuesAtBreakpoints(f) - jnp.array([-1, 0, .5, 1])) < EPS))

def test_source_2():
    f = cj.chebfun(lambda x: x, domain=[-1, 0, .5, 1])
    v = cj.getValuesAtBreakpoints(f, [lambda x: x, lambda x: x, 1])
    assert bool(jnp.all(jnp.abs(v - jnp.array([-1, 0, .5, 1])) < EPS))

def test_source_3():
    f = cj.chebfun(lambda x: x, domain=[-1, 0, .5, 1])
    assert bool(jnp.all(cj.getValuesAtBreakpoints(f, lambda x: x + 100*(x == 0)) == jnp.array([-1, 100, .5, 1])))

def test_source_4():
    f = cj.chebfun(lambda x: x, domain=[-1, 0, .5, 1])
    assert bool(jnp.all(cj.getValuesAtBreakpoints(f, jnp.sign) == jnp.array([-1, 0, 1, 1])))

def test_source_5():
    f = cj.chebfun([jnp.sin, lambda x: 1/jnp.sqrt(1+x), lambda x: x+1], domain=[-2, -1, 0, 1], exps=[0, 0, -.5, 0, 0, 0])
    v = cj.getValuesAtBreakpoints(f)
    expected = jnp.array([jnp.sin(-2.), jnp.inf, 1., 2.])
    idx = jnp.array([0, 2, 3])
    assert jnp.max(jnp.abs(v[idx]-expected[idx])) < 10*EPS*jnp.max(jnp.abs(expected[idx]))
    assert v[1] == jnp.inf

def test_source_6():
    f = cj.chebfun([lambda x: jnp.exp(x)-x, lambda x: .75+jnp.sin(10*x)/jnp.exp(x)], domain=[-2, 0, jnp.inf])
    v = cj.getValuesAtBreakpoints(f)
    expected = jnp.array([jnp.exp(-2.)+2, .875, .75])
    assert jnp.max(jnp.abs(v-expected)) < 1000*EPS*f.vscale


def test_supplemental_native_list_and_stored_override():
    f = cj.chebfun([lambda x: 0*x+2, lambda x: 0*x+4], domain=[-1, 0, 1])
    f = f.set_point_values(jnp.array([90., 91., 92.]))
    expected = jnp.array([2., 3., 4.])
    assert jnp.array_equal(cj.getValuesAtBreakpoints(f), expected)
    assert jnp.array_equal(cj.getValuesAtBreakpoints(f.funs), expected)
    assert jnp.array_equal(cj.getValuesAtBreakpoints(f.funs, [-1, 0, 1], 42), expected)


def test_supplemental_nan_complex_and_empty():
    f = cj.chebfun(lambda x: x + 1j, domain=[-1, 0, 1])
    got = cj.getValuesAtBreakpoints(f, lambda x: jnp.where(x == 0, jnp.nan, x+2j))
    assert jnp.max(jnp.abs(got - jnp.array([-1+2j, 1j, 1+2j]))) < 5*EPS
    assert cj.getValuesAtBreakpoints(cj.chebfun()).size == 0


def test_supplemental_array_valued_nan():
    f = cj.chebfun(lambda x: jnp.stack([x, 2*x], axis=-1), domain=[-1, 0, 1])
    op = lambda x: jnp.stack([jnp.where(x==0, jnp.nan, x+10), 3*x], axis=-1)
    got = cj.getValuesAtBreakpoints(f.funs, [-1, 0, 1], op)
    expected = jnp.array([[9., -3.], [0., 0.], [11., 3.]])
    assert jnp.max(jnp.abs(got-expected)) < 5*EPS


class TestChebfunGetValuesAtBreakpoints:
    def test_identity(self):
        # pass(1)
        f = cj.chebfun(lambda x: x, domain=[-1.0, 0.0, 0.5, 1.0])
        v = np.asarray(cj.getValuesAtBreakpoints(f))
        np.testing.assert_allclose(v, [-1.0, 0.0, 0.5, 1.0],
                                   atol=1e-15)

    def test_custom_op(self):
        # pass(3)-(4)
        f = cj.chebfun(lambda x: x, domain=[-1.0, 0.0, 0.5, 1.0])

        def op(x):
            return x + 100.0 * (x == 0)

        v = np.asarray(cj.getValuesAtBreakpoints(f, op))
        np.testing.assert_allclose(v, [-1.0, 100.0, 0.5, 1.0],
                                   atol=1e-15)
        v2 = np.asarray(cj.getValuesAtBreakpoints(f, jnp.sign))
        np.testing.assert_allclose(v2, [-1.0, 0.0, 1.0, 1.0],
                                   atol=1e-15)


def test_supplemental_coefficient_limits_and_assignment_shape():
    import pytest

    from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
    from chebfunjax.domain import Domain
    coeffs = jnp.array([1e16, 1., -1e16, 3., 2.])
    f = Chebfun(funs=[_Piece.from_coeffs(coeffs, -1., 1.)], domain=Domain((-1., 1.)))
    expected = jnp.array([jnp.sum(coeffs.at[1::2].set(-coeffs[1::2])), jnp.sum(coeffs)])
    assert jnp.array_equal(cj.getValuesAtBreakpoints(f), expected)
    assert jnp.array_equal(cj.getValuesAtBreakpoints(f, lambda x: 42), jnp.array([42., 42.]))
    with pytest.raises(ValueError, match='dimensions'):
        cj.getValuesAtBreakpoints(f, lambda x: jnp.ones((1, 2)))
