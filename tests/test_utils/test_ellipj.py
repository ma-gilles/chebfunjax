"""Independent bounded numerical and staging controls for Jacobi functions."""
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from scipy.special import ellipj as scipy_ellipj

import chebfunjax as cj
from chebfunjax.utils.ellipj import ellipj


@pytest.mark.parametrize('m', [0., 1e-16, 1e-8, .1, .5, .75, .99, 1.])
def test_real_sampled_range(m):
    u = np.linspace(-100, 100, 501)
    got = np.asarray(jax.jit(ellipj)(jnp.asarray(u), m))
    expected = np.asarray(scipy_ellipj(u, m)[:3])
    np.testing.assert_allclose(got, expected, atol=8e-14, rtol=8e-14)


@pytest.mark.parametrize('m', [1e-16, .5, 1-1e-14])
def test_high_precision_real_transition_and_quarter_periods(m):
    with mp.workdps(70):
        period = float(mp.ellipk(m))
        points = [-100., -period, -1., 0., period-1e-8, period, period+1e-8, 100.]
        expected = np.array([[float(mp.ellipfun(kind, u, m)) for u in points]
                             for kind in ('sn', 'cn', 'dn')])
    got = np.asarray(ellipj(jnp.array(points), m))
    np.testing.assert_allclose(got, expected, atol=5e-14, rtol=5e-14)


@pytest.mark.parametrize('m', [0., .25, .75, 1.])
def test_complex_sampled_rectangle(m):
    points = np.array([complex(x, y) for x in [-4., -.3, 0., .7, 4.] for y in [-1., -.2, .4, 1.]])
    with mp.workdps(70):
        expected = np.array([[complex(mp.ellipfun(kind, u, m)) for u in points]
                             for kind in ('sn', 'cn', 'dn')])
    np.testing.assert_allclose(np.asarray(jax.jit(ellipj)(jnp.array(points), m)), expected,
                               atol=5e-14, rtol=5e-14)


@pytest.mark.parametrize('m', [0., .5, .99999999999999, 1.])
def test_u_jvp(m):
    u = jnp.array([-5., -.1, 0., .4, 5.])
    (s, c, d), derivative = jax.jvp(lambda x: ellipj(x, m), (u,), (jnp.ones_like(u),))
    np.testing.assert_allclose(derivative, (c*d, -s*d, -m*s*c), atol=2e-14, rtol=2e-14)


def test_stable_hyperbolic_endpoint():
    u = jnp.array([-1000., -720., 0., 720., 1000.])
    sn, cn, dn = ellipj(u, 1.)
    np.testing.assert_array_equal(sn, [-1., -1., 0., 1., 1.])
    assert np.all(np.isfinite(cn)) and np.all(np.asarray(cn) >= 0)
    np.testing.assert_array_equal(cn, dn)


def test_broadcast_and_invalid_parameter():
    u = jnp.array([-.5, .2])[:, None]
    m = jnp.array([0., .5, 1.])[None, :]
    expected = scipy_ellipj(np.asarray(u), np.asarray(m))[:3]
    np.testing.assert_allclose(ellipj(u, m), expected, atol=1e-15)
    assert np.isnan(np.asarray(jax.jit(ellipj)(u, -1.))).all()
    assert np.isnan(np.asarray(jax.jit(ellipj)(u, 1.1))).all()


def test_public_complex_and_parameter_function():
    x = cj.chebfun(lambda x: x)
    u = .4+.2j*x
    for got, k in zip(cj.ellipj(u, .75), ('sn', 'cn', 'dn')):
        points = np.linspace(-1, 1, 11)
        with mp.workdps(50):
            expected = [complex(mp.ellipfun(k, .4+.2j*y, .75)) for y in points]
        np.testing.assert_allclose(got(jnp.array(points)), expected, atol=2e-14)
    m = .4+.2*x
    for got, expected in zip(cj.ellipj(.3, m), scipy_ellipj(.3, np.array([.2, .4, .6]))[:3]):
        np.testing.assert_allclose(got(jnp.array([-1., 0., 1.])), expected, atol=2e-14)
    for got, expected in zip(cj.ellipj(x, m), scipy_ellipj(np.array([-1., 0., 1.]), np.array([.2, .4, .6]))[:3]):
        np.testing.assert_allclose(got(jnp.array([-1., 0., 1.])), expected, atol=2e-14)


@pytest.mark.parametrize('m', [-.1, 1.1, .2j])
def test_public_rejects_invalid_parameter(m):
    with pytest.raises(ValueError, match='parameter'):
        cj.chebfun(lambda x: x).ellipj(m)


def test_primitive_rejects_complex_parameter():
    with pytest.raises(ValueError, match='real'):
        jax.jit(ellipj)(.2, .3+.1j)
