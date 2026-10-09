"""Source scalar minus dispatch, including the public mean2 result.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df,
@separableApprox/minus.m and @spherefun/plus.m.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._plus import eligible
from chebfunjax.spherefun.spherefun import Spherefun


@pytest.fixture(scope='module')
def sphere():
    f = Spherefun.from_function(lambda lam, theta: 2+jnp.cos(theta))
    assert eligible(f)
    return f


def forbid_resampling(*args, **kwargs):
    raise AssertionError('scalar minus must use source plus, not adaptive resampling')


def check_values(f, scalar, result):
    lam = jnp.linspace(-jnp.pi, jnp.pi, 13)
    theta = jnp.linspace(0, jnp.pi, 11)
    ll, tt = jnp.meshgrid(lam, theta)
    expected = 2+jnp.cos(tt)-scalar
    bound = 1000*jnp.finfo(jnp.float64).eps*max(1., float(f.vscale()))
    assert float(jnp.max(jnp.abs(result(ll, tt)-expected))) < bound


def test_mean_subtraction_keeps_factor_path(sphere, monkeypatch):
    mean = sphere.mean2()
    assert jnp.ndim(mean) == 0
    monkeypatch.setattr(Spherefun, '_binary', forbid_resampling)
    result = sphere-mean
    check_values(sphere, mean, result)
    assert abs(float(result.mean2())) < 1000*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('scalar', [1.25, np.float64(1.25), np.asarray(1.25),
                                   jnp.asarray(1.25, dtype=jnp.float32),
                                   jnp.asarray(1.25, dtype=jnp.float64)])
def test_real_scalar_minus_source(sphere, monkeypatch, scalar):
    monkeypatch.setattr(Spherefun, '_binary', forbid_resampling)
    result = sphere-scalar
    check_values(sphere, scalar, result)
