"""Source newDomain value-state and deltafun.changeMap contracts.

Provenance: @chebfun/newDomain.m, @bndfun/changeMap.m,
@deltafun/changeMap.m; Chebfun 7574c77. No standalone MATLAB test_newDomain
exists at the pinned commit; these assert the source operations directly.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun


@pytest.mark.parametrize("target", [(2., 6.), (2., 3., 7.)])
@pytest.mark.parametrize("transposed", [False, True])
def test_preserves_complex_array_coefficients_orientation_and_point_values(target, transposed):
    f = chebfun(lambda x: jnp.stack((x + 1j*x*x, x*x-2j*x), axis=-1),
                domain=(-1., 0., 1.))
    pv = jnp.asarray([[10+1j, 2], [20+2j, 3], [30+3j, 4]])
    f = f.set_point_values(pv)
    if transposed:
        f = f.T
    original = f.domain.breakpoints
    g = f.new_domain(target)
    expected = (2., 4., 6.) if len(target) == 2 else target
    assert g.domain.breakpoints == expected
    assert g.is_transposed == transposed
    assert bool(jnp.array_equal(g.point_values, pv))
    assert f.domain.breakpoints == original
    for k in range(2):
        assert g.funs[k].tech is f.funs[k].tech
        oldmid = (original[k]+original[k+1])/2
        newmid = (expected[k]+expected[k+1])/2
        assert float(jnp.max(jnp.abs(g.funs[k](jnp.asarray(newmid))-f.funs[k](jnp.asarray(oldmid))))) < 1e-14


def test_delta_locations_map_piecewise_and_magnitudes_orders_survive():
    f = chebfun(lambda x: x, domain=(-1., 0., 1.))
    rows = ((-1., 2.), (0., 3., 1), (.5, 4+2j, 2), (1., -5.))
    f = Chebfun(f.funs, f.domain, deltas=rows)
    g = f.new_domain((2., 3., 7.))
    assert g.deltas == ((2., 2.), (3., 3., 1), (5., 4+2j, 2), (7., -5.))
    assert f.deltas == rows


def test_linear_rescaling_keeps_delta_magnitude_as_source():
    f = chebfun(lambda x: x, domain=(-1., 1.))
    f = Chebfun(f.funs, f.domain, deltas=((.5, 2.),))
    assert f.new_domain((0., 8.)).deltas == ((6., 2.),)


def test_source_inconsistent_interval_identifier():
    f = chebfun(lambda x: x, domain=(-1., 0., 1.))
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:newDomain:numints'):
        f.new_domain((0., 1., 2., 3.))
