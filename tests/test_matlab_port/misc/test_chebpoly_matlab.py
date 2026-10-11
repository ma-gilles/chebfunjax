"""Literal CHEBPOLY predicates and source parsing controls.

Provenance
----------
MATLAB source: tests/misc/test_chebpoly.m, chebpoly.m.
Chebfun commit: 7574c77. Native original bound 1e-14 is unchanged.
"""
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebpoly

TOL = 1e-14

def test_native01():
    n = jnp.array([0, 1, 2, 3, 5, 8])
    x = jnp.linspace(-1, 1, 10)[jnp.array([1, 8])]
    actual = chebpoly(n, 1)(x)
    expected = jnp.cos(jnp.arccos(x)[:, None] * n)
    assert jnp.max(jnp.abs(expected - actual)) < TOL

def test_native02():
    n = jnp.array([0, 1, 2, 3, 5, 8])
    x = jnp.linspace(-1, 1, 10)[jnp.array([1, 8])]
    actual = chebpoly(n, 2)(x)
    expected = jnp.sin(jnp.arccos(x)[:, None] * (n + 1)) / jnp.sin(jnp.arccos(x))[:, None]
    assert jnp.max(jnp.abs(expected - actual)) < TOL

def test_native03():
    assert abs(chebpoly(27).max()[1] - 1) < TOL

def test_native04():
    assert abs(chebpoly(44).min()[1] + 1) < TOL

def test_native05():
    v = chebpoly([7, 8]) @ jnp.array([1.0, 1.0])
    assert abs(v(-1.0)) < TOL

def test_native06():
    assert abs(chebpoly(100000)(1.0) - 1) < TOL

def test_column_orientation():
    f = chebpoly(jnp.array([[0], [2]]))
    assert f.is_transposed and f.n_columns == 2

def test_empty():
    assert chebpoly([]).isempty()

def test_noninteger():
    for degree in (-1, 1.5, float('nan')):
        try:
            chebpoly(degree)
        except ValueError as e:
            assert 'integern' in str(e)
        else:
            raise AssertionError('invalid degree accepted')

def test_second_argument_kind():
    assert abs(chebpoly(3, 2)(0.25) - (8 * 0.25 ** 3 - 4 * 0.25)) < TOL
    for k in (0, 2.5, 3):
        try:
            chebpoly(3, k)
        except ValueError as e:
            assert 'kind' in str(e)
        else:
            raise AssertionError('invalid kind accepted')

def test_domain_breakpoints():
    f = chebpoly(3, [-2.0, 0.0, 3.0])
    assert len(f.funs) == 2
    x = jnp.array([-2.0, 0.0, 3.0])
    z = (2 * x - 1) / 5
    assert jnp.max(jnp.abs(f(x) - (4 * z ** 3 - 3 * z))) < TOL
    try:
        chebpoly(2, [-1.0, jnp.inf])
    except ValueError as e:
        assert 'infdomain' in str(e)
    else:
        raise AssertionError('unbounded domain accepted')

def test_preferred_domain():
    from chebfunjax.chebpref import ChebfunPref
    original = ChebfunPref()
    try:
        ChebfunPref.setDefaults('domain', [-2.0, 3.0])
        f = chebpoly(1)
        u = chebpoly(1, 2)
        assert abs(f(-2.0) + 1) < TOL and abs(f(3.0) - 1) < TOL
        assert abs(u(-2.0) + 2) < TOL and abs(u(3.0) - 2) < TOL
    finally:
        ChebfunPref.setDefaults(original)
