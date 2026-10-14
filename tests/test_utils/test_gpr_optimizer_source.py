"""Source parser edge cases for gpr.m, Chebfun commit 7574c77.

Native R2025b captures establish NaN likelihood construction failure for
zero data with default sigma/L, and invalid optimizer domain for no data.
A supplied L bypasses optimization, so positive observation noise still
allows zero data. No selected-L tolerance is inferred from these cases.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.gpr import LinAlgError, gpr, gpr_chebfun

jax.config.update('jax_enable_x64', True)

@pytest.mark.parametrize('api', [gpr, gpr_chebfun])
@pytest.mark.parametrize('x,y,kwargs,error', [
    ([], [], {}, ValueError),
    ([0., 1.], [0., 0.], {}, ValueError),
    ([0., 1.], [0., 0.], {'length_scale': .1}, LinAlgError),
])
def test_source_parser_failure_branches(api, x, y, kwargs, error):
    with pytest.raises(error):
        api(x, y, **kwargs)

@pytest.mark.parametrize('api', [gpr, gpr_chebfun])
def test_source_zero_data_with_fixed_scale_and_noise(api):
    result = api([0., 1.], [0., 0.], length_scale=.1, noise=.1)
    values = (result['mean'] if api is gpr
              else result[0](jnp.asarray([0., .5, 1.])))
    assert bool(jnp.all(values == 0))
