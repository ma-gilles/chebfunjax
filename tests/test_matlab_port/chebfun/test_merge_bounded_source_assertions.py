"""Port twelve bounded assertions from MATLAB tests/chebfun/test_merge.m.

Assertions1-11,14 retain source inputs and bounds. Assertions12,13 require
singular/unbounded adapters and remain an explicit separate coverage gap.
Python selective merge takes locations; removed indices are recovered from
exact before/after domains to check the MATLAB mergedPts assertion.

Provenance
----------
MATLAB source : tests/chebfun/test_merge.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

import chebfunjax as cj

EPS = float(jnp.finfo(jnp.float64).eps)


def _removed(f, g):
    return [k + 1 for k, x in enumerate(f.domain.breakpoints)
            if x not in g.domain.breakpoints]


def test_source_assertions_1_2_easy_square():
    f = cj.chebfun(lambda x: x**2, domain=[-1., 0., 1.], splitting=True)
    g = f.merge()
    assert len(g.funs) == 1
    xx = jnp.linspace(-1., 1., 100)
    assert float(jnp.max(jnp.abs(f(xx) - g(xx)))) < 10*EPS


def test_source_assertions_3_4_5_selective_merge():
    f = cj.chebfun(lambda x: jnp.sin(10*jnp.pi*x),
                   domain=[-1., -.5, 0., .5, 1., 1.5, 2.], splitting=True)
    g = f.merge(index=[-.5, .5, 1., 1.5])
    assert len(g.funs) == 2
    xx = jnp.linspace(-1., 2., 100)
    assert float(jnp.max(jnp.abs(f(xx) - g(xx)))) < 1e2*EPS
    assert _removed(f, g) == [2, 4, 5, 6]


def test_source_assertions_6_7_8_nonsmooth_merge():
    h = cj.chebfun(lambda x: jnp.sin(x) + jnp.abs(x-.5),
                   domain=[-1., 0., 1.], splitting=True)
    p = h.merge()
    assert len(p.funs) == 2
    assert float(jnp.max(jnp.abs(jnp.array(p.domain.breakpoints)
                               - jnp.array([-1., .5, 1.])))) < 10*EPS
    xx = jnp.linspace(-1., 2., 100)
    assert float(jnp.max(jnp.abs(h(xx) - p(xx)))) < 10*EPS
    assert _removed(h, p) == [2]


def test_source_assertion_9_array_valued_close_breaks():
    f = cj.chebfun(lambda x: jnp.stack([x, x**2], axis=-1),
                   domain=[-1., -.1, -.1+EPS, 0., 1.])
    g = f.merge()
    assert g.domain.breakpoints == (-1., 1.)


def test_source_assertion_10_smooth_row():
    f = cj.chebfun(lambda x: x, domain=[-1., 0., 1.])
    g = f.transpose().merge()
    assert g.domain.breakpoints == (-1., 1.) and g.is_transposed


def test_source_assertion_11_nonsmooth_row():
    f = cj.chebfun(jnp.abs, domain=[-1., 0., 1.])
    g = f.transpose().merge()
    assert g.domain.breakpoints == (-1., 0., 1.) and g.is_transposed


def test_source_assertion_14_vertical_scale_invariance():
    op = lambda x: 1/(1+25*x**2) + 1e-14*jnp.sign(x)
    g = cj.chebfun(op, domain=[-1., 0., 1.])
    h = cj.chebfun(lambda x: 100*op(x), domain=[-1., 0., 1.])
    assert len(g.merge().funs) == len(h.merge().funs)
