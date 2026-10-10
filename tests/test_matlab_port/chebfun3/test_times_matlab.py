"""Original MATLAB Chebfun tests/chebfun3/test_times.m assertions.

Provenance
----------
MATLAB source : tests/chebfun3/test_times.m
Chebfun commit: 7574c77
Original functions, iteration counts, domains, norm predicates and bounds retained.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1e4 * ChebfunPref().cheb3Prefs.chebfun3eps

@pytest.fixture(scope='module')
def times_fields():
    f = chebfun3(lambda x, y, z: jnp.cos(x*y*z))
    h = chebfun3(lambda x, y, z: 2*jnp.cos(x*y*z))
    k = chebfun3(lambda x, y, z: jnp.cos(x*y*z)**2)
    return f, h, k

@pytest.mark.parametrize('statement', range(1, 7))
def test_native_times_first_six(times_fields, statement):
    f, h, k = times_fields
    # Python * adapts native elementwise and scalar-matrix multiplication.
    if statement in (1, 2):
        error = f*2-h
    elif statement in (3, 4):
        error = 2*f-h
    elif statement == 5:
        error = f**2-k
    else:
        error = f*f-k
    assert error.norm() < TOL

@pytest.mark.parametrize('domain', [(-1., 1., -1., 1., -1., 1.),
                                   (-2., 2., -2., 2., -2., 2.),
                                   (0., jnp.pi, 0., jnp.pi, -jnp.pi/2, jnp.pi/2)])
def test_native_times_remaining_three(domain):
    ff = lambda x, y, z: jnp.cos(x*y*z)
    gg = lambda x, y, z: x+y+z+x*y*z
    f, g = chebfun3(ff, domain=domain), chebfun3(gg, domain=domain)
    exact = chebfun3(lambda x, y, z: ff(x, y, z)*gg(x, y, z), domain=domain)
    assert (f*g-exact).norm() < jnp.max(jnp.abs(jnp.asarray(domain)))*TOL
