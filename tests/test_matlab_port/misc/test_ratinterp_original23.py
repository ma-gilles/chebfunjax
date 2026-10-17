"""Literal default tests/misc/test_ratinterp.m, pinned7574c77.

Explicit outputs preserve MATLAB output counts; the retained older tests use
its separate Python compatibility tuple. Native deterministic grid words retain
column/row operand storage. There are no random draws in the original file.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain
from chebfunjax.utils.ratapprox import ratinterp


@pytest.fixture(autouse=True)
def factory_preferences():
    saved = ChebfunPref()
    ChebfunPref.setDefaults("factory")
    try:
        yield
    finally:
        ChebfunPref.setDefaults(saved)


def grid(name):
    data = json.loads((Path(__file__).parent / "fixtures" /
                       "ratinterp_original23_native_grids.json").read_text())[name]
    def component(key):
        return jnp.asarray([int(word, 16) for word in data[key]],
                           dtype=jnp.uint64).view(jnp.float64)
    real, imag = component("real_words"), component("imag_words")
    values = jax.lax.complex(real, imag) if data["complex_storage"] else real
    return values.reshape(tuple(data["shape"]), order="F")


def rational(x):
    return (x**4 - 3) / ((x + .2) * (x - 2.2))


def roots_predicate(mu, nu, poles):
    assert mu == 4 and nu == 2
    assert jnp.max(jnp.abs(jnp.sort(jnp.asarray(poles).reshape(-1)) -
                           jnp.asarray([-.2, 2.2]))) < 1e-10


@pytest.mark.parametrize("clause", range(1, 24))
def test_original_predicate(clause):
    if clause == 1:
        p, q, r, mu, nu, poles, res = ratinterp(
            rational, 10, 10, None, "type0", domain=(0., 2.), outputs=7)
        roots_predicate(mu, nu, poles)
    elif clause <= 4:
        kind = {2: "type1", 3: "type2", 4: "equi"}[clause]
        p, q, r, mu, nu = ratinterp(rational, 10, 10, None, kind,
                                   domain=(0., 2.), outputs=5)
        roots_predicate(mu, nu, q.roots(all_roots=True))
    elif clause <= 7:
        x = chebfun(lambda x: x, domain=(0., 2.))
        cf = rational(x)
        kind = {5: "type1", 6: "type2", 7: "equi"}[clause]
        p, q, r, mu, nu = ratinterp(cf, 10, 10, None, kind, outputs=5)
        roots_predicate(mu, nu, q.roots(all_roots=True))
    elif clause == 8:
        p, q, r, mu, nu, poles = ratinterp(
            rational(grid("circle100")), 10, 10, 100, "type0",
            domain=(0., 2.), outputs=6)
        roots_predicate(mu, nu, poles)
    elif clause <= 11:
        name, kind = {9: ("kind1_100", "type1"),
                      10: ("kind2_100", "type2"),
                      11: ("equi100", "equi")}[clause]
        p, q, r, mu, nu = ratinterp(rational(grid(name)), 10, 10, 100,
                                   kind, domain=(0., 2.), outputs=5)
        roots_predicate(mu, nu, q.roots(all_roots=True))
    elif clause == 12:
        p, q, r, mu, nu = ratinterp(lambda x: 1/(x-.2), 10, 10, None,
                                   "type2", domain=(-1., 1.), outputs=5)
        assert mu == 0 and nu == 1
        roots = jnp.asarray(q.roots()).reshape(-1)
        assert roots.size == 1 and jnp.abs(roots[0]-.2) < 1e-10
    elif clause <= 17:
        x = chebfun(lambda x: x, domain=(1., 3.))
        f = abs(x.exp() - 5)
        nodes = grid("arbitrary6") if clause <= 15 else "type2"
        p, q, r = ratinterp(f, 2, 3, None, nodes, outputs=3)
        if clause == 13:
            assert len(p) == 3 and len(q) == 4
        elif clause in (14, 16):
            assert (f - p/q).norm(jnp.inf) < .6
        else:
            xx = grid("xx300")
            # Normalize only evaluation layout to the same native scalar-vector
            # operation; do not broadcast a singleton column across a row.
            actual, expected = jnp.asarray(f(xx)).reshape(-1), jnp.asarray(r(xx)).reshape(-1)
            assert actual.size == expected.size == xx.size
            assert jnp.max(jnp.abs(actual-expected)) < .6
    elif clause == 18:
        p, q, r, mu, nu, poles = ratinterp(lambda x: 1/(x-1.7), 128, 1,
                                          None, "type0", 1e-14, outputs=6)
        assert nu == 0
    elif clause == 19:
        p, q, r, mu, nu, poles = ratinterp(lambda x: jnp.sin(x)/(x-.1),
                                          30, 1, None, grid("complex32"),
                                          1e-14, outputs=6)
        assert poles.size == 1 and jnp.abs(poles[0]-.1) < 1e-10
    elif clause == 20:
        p1, q1 = ratinterp(jnp.exp, 1, 1, domain=(-1., 1.), outputs=2)
        p2, q2 = ratinterp(jnp.exp, 1, 1, domain=Domain([-1., 1.]), outputs=2)
        assert (p1-p2).norm() == 0 and (q1-q2).norm() == 0
    elif clause <= 22:
        # Original21 and22 deliberately repeat the same public invocation.
        def f(x):
            return 1/(x-.2)
        p, q, r = ratinterp(f, 10, 10, None, "type2", 0,
                            domain=(-1., 1.), outputs=3)
        xx = grid("xx20")
        err = jnp.linalg.norm(jnp.asarray(r(xx)).reshape(-1)-f(xx).reshape(-1))
        assert err < 1e-10
    else:
        p, q, r = ratinterp(jnp.ones((5, 1)), 1, 1, None,
                            grid("arbitrary5"), outputs=3)
        assert jnp.abs(r(1)-1) < 1e-10
