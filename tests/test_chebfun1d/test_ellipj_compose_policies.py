"""Source tolerance, point values and overlapping composition policies.

Provenance
----------
MATLAB source: @chebfun/ellipj.m, @chebfun/compose.m, @chebtech/compose.m.
Chebfun commit: 7574c77
AGM formula checks independently specify zero and one-step stopping;
these are builtin-adapter controls, not freshly captured MATLAB outputs.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref


def test_numeric_tolerance_retains_first_terminal_correction():
    u = .7
    mean = (1+math.sqrt(.5))/2
    difference = (1-math.sqrt(.5))/2
    phi = mean*u + .5*math.asin(difference/mean*math.sin(2*mean*u))
    expected = [math.sin(phi), math.cos(phi), math.sqrt(1-.5*math.sin(phi)**2)]
    np.testing.assert_allclose(cj.ellipj(u, .5, .2), expected, atol=3e-16, rtol=0)
    assert abs(float(cj.ellipj(u, .5, .2)[0]-cj.ellipj(u, .5)[0])) > 1e-5


def test_numeric_tolerance_no_iterations_and_jit():
    u = jnp.array([.2, .7])
    got = jax.jit(lambda x, tol: cj.ellipj(x, .5, tol))(u, 1.)
    np.testing.assert_allclose(got, [np.sin(u), np.cos(u), np.sqrt(1-.5*np.sin(u)**2)],
                               atol=3e-16, rtol=0)


def test_broadcast_global_mean_with_different_stop_levels():
    u = .7
    m = jnp.array([.01, .5])
    # Both columns advance once because the second column exceeds tolerance.
    # The first needs no descent but still uses the final arithmetic mean.
    expected = math.sin((1+math.sqrt(.99))/2*u)
    assert abs(float(cj.ellipj(u, m, .2)[0][0])-expected) < 3e-16


@pytest.mark.parametrize('tol', [-1., math.nan, math.inf, .1j, [1., 2.]])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError, match='tolerance'):
        cj.ellipj(.3, .5, tol)


@pytest.mark.parametrize('m', [-.0001, 1.0001])
def test_source_fudge_maps_both_sides_to_zero(m):
    parameter = cj.chebfun(lambda x: 0*x+m)
    expected = [math.sin(.3), math.cos(.3), 1.]
    for got, value in zip(cj.ellipj(.3, parameter, .001), expected):
        np.testing.assert_allclose(got(jnp.array([-.7, .2, .9])), value, atol=3e-16, rtol=0)
    with pytest.raises(ValueError, match='parameter'):
        cj.ellipj(cj.chebfun(lambda x: x), m, .001)


def test_numeric_tolerance_preserves_default_composition_accuracy():
    u = cj.chebfun(lambda x: x, domain=(-1., 0., 1.))
    sites = jnp.linspace(-1, 1, 31)
    for got, expected in zip(cj.ellipj(u, .9, .01), cj.ellipj(sites, .9, .01)):
        assert max(abs(got(sites)-expected)) < 100*got.vscale*np.finfo(float).eps


def test_preference_tolerance_changes_output_approximation():
    u = cj.chebfun(lambda x: 3*x)
    pref = ChebfunPref(chebfuneps=1e-3)
    accurate = cj.ellipj(u, .9, 1e-3)
    coarse = cj.ellipj(u, .9, pref)
    assert all(len(c.funs[0].tech.coeffs) < len(a.funs[0].tech.coeffs)
               for c, a in zip(coarse, accurate))
    x = jnp.linspace(-1, 1, 101)
    for got, expected in zip(coarse, cj.ellipj(3*x, .9, 1e-3)):
        assert max(abs(got(x)-expected)) < 10*pref.chebfuneps


def test_unary_composition_retains_breaks_points_and_orientation():
    f = cj.chebfun(lambda x: x*x, domain=(-1., 0., 1.)).T
    f = f.set_point_values(jnp.array([1., 7., 1.]))
    g = f.compose(jnp.exp)
    assert g.domain == f.domain and g.is_transposed
    np.testing.assert_allclose(g.point_values, jnp.exp(f.point_values), atol=0, rtol=0)
    np.testing.assert_allclose(g(jnp.array([-.3, .2])), np.exp(np.array([.09, .04])), atol=2e-15)


def test_binary_composition_overlap_and_scalar_column_broadcast():
    f = cj.chebfun(lambda x: x, domain=(-1., -.25, 1.))
    g = cj.chebfun(lambda x: jnp.stack((x*x, 1+x), axis=-1), domain=(-1., .5, 1.))
    out = f.compose(lambda a, b: a+b, g)
    assert tuple(out.domain.breakpoints) == (-1., -.25, .5, 1.)
    x = jnp.array([-.7, -.1, .2, .8])
    np.testing.assert_allclose(out(x), np.stack((x+x*x, 1+2*x), axis=-1), atol=2e-15)
    np.testing.assert_allclose(out.point_values, f(jnp.array(out.domain.breakpoints))[:, None]+g(jnp.array(out.domain.breakpoints)), atol=2e-15)


def test_binary_overlap_preserves_original_point_values_and_rows():
    f = cj.chebfun(lambda x: x, domain=(-1., 0., 1.))
    f = f.set_point_values(jnp.array([-1., 7., 1.])).T
    g = cj.chebfun(lambda x: x*x, domain=(-1., .5, 1.))
    g = g.set_point_values(jnp.array([1., 9., 1.])).T
    out = f.compose(lambda a, b: a+b, g)
    assert out.is_transposed
    np.testing.assert_allclose(out.point_values, [0., 7., 9.5, 2.], atol=1e-15)
    with pytest.raises(ValueError, match='row and column'):
        f.compose(lambda a, b: a+b, g.T)
