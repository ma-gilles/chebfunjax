"""Explicit-Z string API: native aaa.m parseInputs, pinned7574c77.

Original16's random query is not captured. These are independent deterministic
controls; the large-grid control preserves its sample grid and exact equality.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.utils.aaa import aaa


def assert_same(expression, callback, z, queries, **options):
    expected = aaa(callback, z, **options)
    actual = aaa(expression, z, **options)
    for a, b in zip(actual[1:], expected[1:]):
        assert bool(jnp.array_equal(a, b))
    ar = actual[0] if isinstance(actual[0], list) else [actual[0]]
    er = expected[0] if isinstance(expected[0], list) else [expected[0]]
    for a, b in zip(ar, er):
        assert bool(jnp.array_equal(a(queries), b(queries)))
        for query in queries:
            assert bool(a(query) == b(query))


@pytest.mark.parametrize('expression,callback', [
    ('abs(x)', jnp.abs),
    ('sin(t) + t.^2', lambda t: jnp.sin(t) + t**2),
    ('exp(x)./(2+x.*x)', lambda x: jnp.exp(x)/(2+x*x)),
    ('1e-3 + x', lambda x: 1e-3+x),
])
def test_explicit_expression(expression, callback):
    # Nondefault full sample domain is retained; no [-1,1] clipping.
    z = jnp.linspace(-1.5, 2.0, 129)
    assert_same(expression, callback, z, jnp.array([-1.25, 0.0, 0.37, 1.75]))


def test_complex_sample_expression():
    z = jnp.linspace(-1, 1, 97) + 0.3j
    assert_same('exp(z)./(2+z.^2)', lambda z: jnp.exp(z)/(2+z**2),
                z, jnp.array([0.1+0.2j, -0.2+0.5j]))


def test_derivative_forwarding():
    assert_same('exp(t)', jnp.exp, jnp.linspace(-1, 1, 65),
                jnp.array([-0.3, 0.1, 0.7]), deriv_deg=2)


def test_omitted_samples_preserve_rejection():
    with pytest.raises(ValueError, match='Z may only be omitted'):
        aaa('abs(x)')


def test_original16_grid_independent_queries():
    z = jnp.linspace(-1.0, 1.0, 10001)
    assert_same('abs(x)', jnp.abs, z,
                jnp.array([-1.0, -0.731, -0.01, 0.0, 0.123, 0.82, 1.0]))
