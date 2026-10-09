"""Source-derived zeroFun controls from Chebfun7574c77 @chebtech/roots.m.

These cover the option contract; injected subdivision tests only propagation,
not numerical subdivision accuracy or a native MATLAB execution comparison.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import chebtech as module
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("coeffs", [[0.], [3.], [0., 0., 0.]])
def test_zero_constant_source_branches(tech, enabled, coeffs):
    f = tech(coeffs=jnp.asarray(coeffs))
    expected = [0.] if enabled and not any(coeffs) else []
    np.testing.assert_array_equal(f.roots(zero_fun=enabled), expected)
    if enabled:
        np.testing.assert_array_equal(f.roots(), expected)


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("enabled", [False, True])
def test_array_zero_column_padding(tech, enabled):
    f = tech(coeffs=jnp.asarray([[0., 3., -.25], [0., 0., 1.]]))
    expected = [[0. if enabled else np.nan, np.nan, .25]]
    np.testing.assert_array_equal(f.roots(zero_fun=enabled), expected)


@pytest.mark.parametrize("enabled", [False, True])
def test_coefficient_only_zero_compatibility(enabled):
    result = module._roots_colleague(jnp.zeros(5), zero_fun=enabled)
    np.testing.assert_array_equal(result, [0.] if enabled else [])


@pytest.mark.parametrize("enabled", [False, True])
def test_all_trimmed_leaf(enabled):
    result = module._roots_main(np.zeros(5), 100*np.finfo(float).eps,
                                zero_fun=enabled)
    np.testing.assert_array_equal(result, [0.] if enabled else [])


@pytest.mark.parametrize("enabled", [False, True])
def test_recursive_zero_option_propagates(monkeypatch, enabled):
    calls = []

    def subdivision(coeffs):
        calls.append(len(coeffs))
        return jnp.zeros(3), jnp.zeros(3)

    monkeypatch.setattr(module, "_roots_subdivide", subdivision)
    c = np.zeros(51)
    c[-1] = 1.
    result = module._roots_main(c, 100*np.finfo(float).eps, zero_fun=enabled)
    split = -.004849834917525
    expected = [(split-1)/2, (split+1)/2] if enabled else []
    np.testing.assert_array_equal(result, expected)
    assert calls == [51]


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
@pytest.mark.parametrize("enabled", [False, True])
def test_nonzero_linear_and_all_roots_unaffected(tech, enabled):
    f = tech(coeffs=jnp.asarray([-2., 1.]))
    np.testing.assert_array_equal(f.roots(zero_fun=enabled), [])
    np.testing.assert_array_equal(f.roots(all_roots=True, zero_fun=enabled), [2.])
