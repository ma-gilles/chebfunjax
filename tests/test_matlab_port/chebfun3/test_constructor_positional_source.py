"""Original positional syntax and source-order Chebfun3 parser controls.

Provenance: tests/chebfun3/test_construnctorsyntax.m and
@chebfun3/constructor.m, Chebfun 7574c77.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.utils.quadrature import chebpts3


def fn(x, y, z):
    return jnp.sin(x*y*z)


@pytest.mark.parametrize("order", ["length", "length_domain", "domain_length"])
def test_original_fixed_lengths(order):
    dom = (-2,1,-2,1,-2,1)
    ns = [10,15,5]
    if order == "length":
        f = chebfun3(fn, ns)
        assert f.domain == (-1.,1.,-1.,1.,-1.,1.)
    elif order == "length_domain":
        f = chebfun3(fn, ns, dom)
        assert f.domain == dom
    else:
        f = chebfun3(fn, dom, ns)
        assert f.domain == dom
    assert f.length() == (10,15,5)


def test_original_split_domain_is_ignored_by_native_parser():
    f = chebfun3(lambda x,y,z: jnp.cos(x)+jnp.sin(x*y*z), [-1,0], [2,3])
    assert f.domain == (-1.,1.,-1.,1.,-1.,1.)


def test_original_existing_object_selectors():
    f = chebfun3(fn, [-2,1,-2,1,-2,1])
    assert chebfun3(f, "classic") is f
    assert chebfun3(f, "chebfun3f") is f


def test_positional_sampling_uses_domain_at_argument_position():
    ns = [4,5,6]
    dom = (-2.,1.,0.,2.,3.,4.)
    callback = lambda x,y,z: x+2*y+3*z
    before = chebfun3(callback, ns, dom)
    after = chebfun3(callback, dom, ns)
    target = chebpts3(*ns, domain=dom)
    original = chebpts3(*ns)
    np.testing.assert_allclose(before(*target), callback(*original), atol=1e-12, rtol=0)
    np.testing.assert_allclose(after(*target), callback(*target), atol=1e-12, rtol=0)


@pytest.mark.parametrize("route", ["rank", "rank_domain", "domain_rank"])
def test_original_rank_selectors(route):
    dom = (-2.,1.,-2.,1.,-2.,1.) if route != "rank" else (-1.,1.,-1.,1.,-1.,1.)
    baseline = chebfun3(fn, domain=dom)
    old = baseline.rank
    if route == "rank":
        result = chebfun3(fn, "rank", [v-1 for v in old])
    elif route == "rank_domain":
        result = chebfun3(fn, "rank", [v-1 for v in old], dom)
    else:
        result = chebfun3(fn, dom, "rank", [5,6,7])
    assert all(new < prior for new,prior in zip(result.rank, old))
    assert result.domain == dom
