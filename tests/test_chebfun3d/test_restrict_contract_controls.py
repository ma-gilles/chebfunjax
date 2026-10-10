"""Added factor-preservation controls, separate from native restrict ten."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech

DOMAIN = (-1., 1.)*3


def field(ranks=(2, 3, 4), complex_core=True):
    factors = [[Chebtech2.from_coeffs(jnp.eye(r)[i]) for i in range(r)] for r in ranks]
    core = jnp.arange(1, 1+ranks[0]*ranks[1]*ranks[2], dtype=jnp.float64).reshape(ranks)/100
    if complex_core:
        core = core+1j*core[::-1, ::-1, ::-1]
    return Chebfun3(*factors, core, DOMAIN)


@pytest.mark.parametrize('fixed', [(True, True, True), (False, True, True),
                                 (True, False, True), (True, True, False),
                                 (True, False, False), (False, True, False),
                                 (False, False, True), (False, False, False)])
def test_complex_all_restrictions_without_resampling(monkeypatch, fixed):
    f = field()
    monkeypatch.setattr(Chebfun3, 'from_function', lambda *a, **k: pytest.fail('3D resampling'))
    monkeypatch.setattr(Chebfun2, 'from_function', lambda *a, **k: pytest.fail('2D resampling'))
    d = tuple(v for axis in fixed for v in ((.2, .2) if axis else (-.5, .75)))
    result = f.restrict(d)
    xs = jnp.asarray([-.4, .1, .6])
    xyz = [jnp.full_like(xs, .2) if flag else xs for flag in fixed]
    free = [xyz[k] for k in range(3) if not fixed[k]]
    observed = result(*free) if free else result
    if len(free) == 1:
        assert observed.shape == xs.shape
    assert jnp.max(jnp.abs(jnp.asarray(observed).reshape(-1)-f(*xyz).reshape(-1))) < 2e-13
    if len(free) == 3:
        assert result.core is f.core
        assert result.domain == d


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_free_axis_domain_rejection(axis):
    d = [0.]*6
    d[2*axis:2*axis+2] = [-2., .5]
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:restrict:subdom'):
        field().restrict(d)


def test_fixed_coordinate_evaluates_outside():
    f = field()
    assert jnp.abs(f.restrict((2., 2., .1, .1, .3, .3))-f(2., .1, .3)) == 0


@pytest.mark.parametrize('domain,message', [([0.]*5, 'Domain not determined'),
                                           ([0.]*7, 'Domain not determined'),
                                           ('bad', 'Unrecognizable domain'),
                                           ([1j]*6, 'Unrecognizable domain')])
def test_invalid_domain(domain, message):
    with pytest.raises(ValueError, match=message):
        field().restrict(domain)


@pytest.mark.parametrize('zero', [False, True])
def test_rank_one_cuboid(zero):
    f = field((1, 1, 1), False)
    if zero:
        f = Chebfun3(f.cols, f.rows, f.tubes, jnp.zeros((1, 1, 1)), DOMAIN)
    g = f.restrict((-.5, .5)*3)
    assert g.core is f.core
    assert jnp.abs(g(.1, .2, .3)-f(.1, .2, .3)) < 1e-15


def test_periodic_free_axes_use_chebfun_restriction():
    t = Trigtech.from_coeffs(jnp.asarray([1.+0j]), is_real=True)
    f = Chebfun3([t], [t], [t], jnp.ones((1, 1, 1)), DOMAIN)
    g = f.restrict((-.5, .5)*3)
    assert all(isinstance(x, Chebtech2) for mode in (g.cols, g.rows, g.tubes) for x in mode)
    assert g.core is f.core
    assert jnp.abs(g(.1, .2, .3)-1.) < 1e-14
