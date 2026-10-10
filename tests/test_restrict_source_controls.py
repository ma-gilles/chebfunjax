"""Controls for source restriction branches absent from the29 source slots."""
import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import tweak_domain


@pytest.mark.parametrize('domain', [[], [.2], [.2, .2]])
def test_empty_subdomain(domain):
    assert cj.chebfun(jnp.sin).restrict(domain).isempty()


def test_breakpoint_values_orientation_and_duplicates():
    f = cj.chebfun(lambda x: x+1j*x, domain=[-1, 0, 1]).set_point_values(jnp.array([2j, 3j, 4j])).T
    g = f.restrict([-.5, 0, 0, .25, 1])
    assert g.is_transposed
    assert g.domain.breakpoints == (-.5, 0., .25, 1.)
    assert complex(g(0)) == 3j
    assert complex(g(1)) == 4j


def test_narrow_interval_and_eval_jit_ad():
    f = cj.chebfun(lambda x: x*x)
    g = f.restrict([.1, .1+1e-12])
    x = .1+4e-13
    assert abs(float(jax.jit(lambda t: g(t))(x))-x*x) < 1e-15
    assert abs(float(jax.grad(lambda t: g(t))(x))-2*x) < 1e-4


def test_outside_beyond_source_tweak_threshold():
    f = cj.chebfun(jnp.sin)
    with pytest.raises(ValueError, match='restrict:subdom'):
        f.restrict([-1-1e-14, .5])


def test_periodic_row_orientation():
    f = cj.chebfun(lambda x: jnp.sin(jnp.pi*x), trig=True).T
    assert f.restrict(f.domain).is_transposed
    assert f.restrict([-.5, .2]).is_transposed


def test_finite_union_keeps_distinct_close_breaks():
    f = cj.chebfun(lambda x: x, domain=[-1, 0, 1])
    g = cj.chebfun(lambda x: x*x, domain=[-1, 1e-14, 1])
    a, b = type(f)._overlap(f, g)
    assert a.domain.breakpoints == b.domain.breakpoints == (-1., 0., 1e-14, 1.)


def test_finite_union_tweaks_source_nearby_breaks():
    f = cj.chebfun(lambda x: x, domain=[-1, 0, 1])
    g = cj.chebfun(lambda x: x*x, domain=[-1, 5e-16, 1])
    a, b = type(f)._overlap(f, g)
    assert a.domain.breakpoints == b.domain.breakpoints == (-1., 0., 1.)


def test_finite_union_tweaks_subnormal_breakpoint_before_restriction():
    f = cj.chebfun(lambda x: x > 0, domain=(-1., 1.), splitting=True)
    g = cj.heaviside(cj.chebfun("x", domain=(-1., 1.)))
    assert 0.0 < f.domain.breakpoints[1] < float.fromhex("0x1p-1022")
    a, b = type(f)._overlap(f, g)
    assert a.domain.breakpoints == b.domain.breakpoints == (-1., 0., 1.)
    assert float((a - b).norm(2)) < 1e-14


def test_tweak_domain_rounds_sub_half_value_before_large_tolerance():
    below_half = float.fromhex("0x1.fffffffffffffp-2")
    just_below = float.fromhex("0x1.ffffffffffffep-2")
    f = cj.chebfun(lambda x: x, domain=(-10., just_below, 10.))
    g = cj.chebfun(lambda x: x*x, domain=(-10., below_half, 10.))
    f2, g2, _, _ = tweak_domain(f, g, tol=0.5)
    assert f2.domain.breakpoints == g2.domain.breakpoints == (-10., 0., 10.)


def test_tweak_domain_preserves_large_exact_odd_integer_rounding():
    odd = float(2**52 + 1)
    f = cj.chebfun(lambda x: x, domain=(-2**54, odd - 1, 2**54))
    g = cj.chebfun(lambda x: x*x, domain=(-2**54, odd + 1, 2**54))
    f2, g2, _, _ = tweak_domain(f, g, tol=3.0)
    assert f2.domain.breakpoints == g2.domain.breakpoints == (-2**54, odd, 2**54)


@pytest.mark.parametrize('right', [False, True])
def test_unbounded_union(right):
    exact = (lambda x: jnp.exp(-x)) if right else jnp.exp
    d1, d2 = ([0, 1, float('inf')], [0, 2, float('inf')]) if right else (
        [-float('inf'), 0, 1], [-float('inf'), -1, 1])
    f, g = cj.chebfun(exact, domain=d1), cj.chebfun(exact, domain=d2)
    a, b = type(f)._overlap(f, g)
    assert a.domain.breakpoints == b.domain.breakpoints == tuple(sorted(set(d1+d2)))
    x = jnp.asarray([.2, 1.3, 8] if right else [-8, -1.3, .2])
    assert float(jnp.max(jnp.abs(a(x)-exact(x)))) < 1e-13
    assert float(jnp.max(jnp.abs(b(x)-exact(x)))) < 1e-13


def test_unbounded_domain_mismatch_rejected():
    f, g = cj.chebfun(jnp.exp, domain=[-float('inf'), 1]), cj.chebfun(jnp.exp, domain=[-float('inf'), 2])
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:overlap:domains'):
        type(f)._overlap(f, g)
