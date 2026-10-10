"""Complex-factor controls for partial reductions.

Provenance
----------
MATLAB source: @chebfun3/{sum,sum2,mean,mean2}.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2


@pytest.mark.parametrize('axes', [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3)])
def test_stored_complex_factors_and_physical_normalization(axes):
    domain = (-2., 1., 0., 4., -3., 2.)
    offsets = (1 + .5j, -.25 + 1j, .5 - .25j)
    slopes = (.25 + .125j, -.125 + .25j, .5 + .125j)
    factors = [Chebtech2.from_coeffs(jnp.asarray([a, b]))
               for a, b in zip(offsets, slopes, strict=True)]
    core = 2 - .5j
    f = Chebfun3(cols=[factors[0]], rows=[factors[1]], tubes=[factors[2]],
                core=jnp.asarray([[[core]]]), domain=domain)
    expected = jnp.asarray(core)
    area = 1.
    points = []
    for index, (a, b) in enumerate(zip(offsets, slopes, strict=True), start=1):
        lo, hi = domain[2*(index-1):2*index]
        if index in axes:
            expected = expected * ((hi-lo)*a)
            area *= hi-lo
        else:
            t = jnp.asarray([-.75, .125, .5])
            points.append((lo+hi)/2 + (hi-lo)/2*t)
            expected = expected * (a + b*t)
    if len(axes) == 1:
        result, average = f.sum(axes[0]), f.mean(axes[0])
    else:
        result, average = f.sum2(axes), f.mean2(axes)
    assert jnp.max(jnp.abs(result(*points) - expected)) < 1e4*jnp.finfo(jnp.float64).eps
    assert jnp.max(jnp.abs(average(*points) - expected/area)) < 1e4*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize('dim', [0, 4, -1])
def test_sum_rejects_invalid_dimension(dim):
    f = Chebfun3.from_function(lambda x, y, z: x+y+z)
    with pytest.raises(ValueError, match='1, 2, or 3'):
        f.sum(dim)


@pytest.mark.parametrize('dims', [(1, 1), (0, 2), (2, 4)])
def test_sum2_rejects_invalid_pair(dims):
    f = Chebfun3.from_function(lambda x, y, z: x+y+z)
    with pytest.raises(ValueError, match='two distinct'):
        f.sum2(dims)
