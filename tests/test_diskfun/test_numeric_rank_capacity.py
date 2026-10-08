"""Numeric disk construction must resolve the complete doubled BMC grid.

Provenance: @diskfun/constructor.m PhaseOne, Chebfun commit7574c77.
The source uses the undoubled row count as a rank cap; this adaptation
uses the actual doubled matrix dimensions and keeps its stopping tolerance.
Analytic harmonic polynomials expose the insufficient old cap independently
of the original Helmholtz k=7 regression. Bound1e-12 is a new correctness
control, not a modified MATLAB bound.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import Diskfun


def _polynomial(theta, radius, degree):
    value = jnp.ones_like(theta + radius)
    for k in range(1, degree+1):
        value = value + .7**k*radius**k*(jnp.cos(k*theta)+.2*jnp.sin(k*theta))
    return value


@pytest.mark.parametrize('nr,nt', [(3,12),(4,16),(5,20)])
def test_doubled_bmc_grid_resolves_all_harmonic_modes(nr,nt):
    theta = -jnp.pi+2*jnp.pi*jnp.arange(nt)/nt
    radius = jnp.cos(jnp.pi*jnp.arange(nr-1,-1,-1)/(2*nr-2))
    values = _polynomial(theta[None,:], radius[:,None], 2*nr-2)
    actual = Diskfun.from_values(values)
    assert float(jnp.max(jnp.abs(actual.fevalm(theta,radius)-values))) < 1e-12
    off_theta = jnp.linspace(-3.,3.,15)
    off_radius = jnp.linspace(0.,1.,11)
    expected = _polynomial(off_theta[None,:],off_radius[:,None],2*nr-2)
    assert float(jnp.max(jnp.abs(actual.fevalm(off_theta,off_radius)-expected))) < 1e-12
