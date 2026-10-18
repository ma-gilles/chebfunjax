"""Continuous L2 normalization of public polynomial minimax status.

The first exchange for t**3+t**2 at degree one has trial .5+t,
levelled error .5, and maximum error37/54. Thus delta is5/27 divided
by the continuous L2 norm, sqrt(12*domain_width/35). Native minimax.m
uses opts.normf=norm(f), not a sampled maximum. Fresh MATLAB controls
are retained in the associated evidence packet.
"""

import math

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.utils.minimax import minimax


@pytest.mark.parametrize('domain,amplitude', [
    ((-1., 1.), 1.), ((-1., 1.), -3.), ((2., 5.), 1.),
])
@pytest.mark.parametrize('input_kind', ['chebfun', 'callable'])
def test_public_first_exchange_uses_continuous_norm(domain, amplitude, input_kind):
    a, b = domain

    def target(x):
        t = (2*x-a-b)/(b-a)
        return amplitude*(t**3+t**2)

    f = chebfun(target, domain=domain) if input_kind == 'chebfun' else target
    with pytest.warns(RuntimeWarning, match='did not converge after 1 iterations'):
        result = minimax(f, 1, domain=domain, max_iter=1)
    normf = abs(amplitude)*math.sqrt(12*(b-a)/35)
    expected_delta = abs(amplitude)*(5/27)/normf
    #128eps permits the actual polynomial construction, integration and
    # first exchange rounding; no inherited solver tolerance is widened.
    roundoff = 128*float(jnp.finfo(jnp.float64).eps)
    assert result.delta == pytest.approx(expected_delta, rel=roundoff, abs=roundoff)
    assert result.err == pytest.approx(abs(amplitude)*37/54, rel=roundoff)
    assert result.iter == 1
    assert jnp.allclose(result.coeffs, amplitude*jnp.asarray([.5, 1.]),
                        rtol=roundoff, atol=roundoff)
    assert jnp.allclose(result.xk, jnp.asarray([a, (a+2*b)/3, b]),
                        rtol=roundoff, atol=roundoff)
