"""Core tests for the pointValues field, the ``turbo`` construction flag, and
trig/cheb mixed-tech casting (Opus 4.8).

These mirror the MATLAB-port tests in ``tests/test_matlab_port/chebfun/`` with
direct (non-golden) assertions on the new ``chebfunjax`` surface:
:meth:`Chebfun.point_values` / :meth:`Chebfun.set_point_values`, the ``turbo``
flag threaded through :func:`chebfun`, and :func:`_cast_tech_pair`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import _cast_tech_pair
from chebfunjax.tech.chebtech import Chebtech2


def _tech(f):
    return type(f.funs[0].tech).__name__


class TestPointValues:
    def test_default_is_endpoint_feval(self):
        f = cj.chebfun(jnp.sin, domain=(-1, 0, 1))
        pv = np.asarray(f.point_values)
        bps = np.array(list(f.domain.breakpoints))
        assert np.allclose(pv, np.sin(bps), atol=1e-13)

    def test_set_and_read_override(self):
        f = cj.chebfun(jnp.sin, domain=(-1, 0, 1))
        vals = jnp.asarray([10.0, 20.0, 30.0])
        g = f.set_point_values(vals)
        assert np.allclose(np.asarray(g.point_values), [10.0, 20.0, 30.0])
        # Original is untouched (immutability).
        assert not np.allclose(np.asarray(f.point_values), [10.0, 20.0, 30.0])

    def test_abs_propagates_override(self):
        f = cj.chebfun(lambda x: x ** 2 + 1.0, domain=(-1, 0, 1))
        g = f.set_point_values(jnp.asarray([-3.0, -4.0, 5.0]))
        assert np.allclose(np.asarray(g.abs().point_values), [3.0, 4.0, 5.0])

    def test_sign_propagates_override(self):
        f = cj.chebfun(lambda x: x ** 2 + 1.0, domain=(-1, 0, 1))
        g = f.set_point_values(jnp.asarray([-3.0, 4.0, 0.0]))
        assert np.allclose(np.asarray(g.sign().point_values), [-1.0, 1.0, 0.0])

    def test_override_does_not_affect_interior_feval(self):
        f = cj.chebfun(jnp.sin, domain=(-1, 1))
        g = f.set_point_values(jnp.asarray([99.0, 99.0]))
        assert abs(float(g(jnp.asarray(0.3))) - float(jnp.sin(jnp.asarray(0.3)))) < 1e-13


class TestTurboFlag:
    def test_adaptive_doubles(self):
        p = cj.chebfun(jnp.exp)
        t = cj.chebfun(jnp.exp, turbo=True)
        assert len(t) == 2 * len(p)

    def test_fixed_length(self):
        assert len(cj.chebfun(jnp.exp, n=40, turbo=True)) == 40

    def test_array_valued_doubles(self):
        av = lambda x: jnp.stack([jnp.exp(x), jnp.cos(x)], axis=-1)  # noqa: E731
        p = cj.chebfun(av)
        t = cj.chebfun(av, turbo=True)
        assert len(t) == 2 * len(p)

    def test_accuracy_preserved(self):
        t = cj.chebfun(jnp.cos, turbo=True)
        xs = jnp.linspace(-1, 1, 41)
        assert float(jnp.max(jnp.abs(t(xs) - jnp.cos(xs)))) < 1e-14


class TestTechCast:
    def test_same_tech_is_identity(self):
        a = Chebtech2.from_function(jnp.exp)
        b = Chebtech2.from_function(jnp.sin)
        ca, cb = _cast_tech_pair(a, b)
        assert ca is a and cb is b

    def test_trig_plus_cheb_casts_to_cheb(self):
        f = cj.chebfun(lambda x: jnp.sin(np.pi * x), domain=(-1, 1), trig=True)
        g = cj.chebfun(lambda x: x, domain=(-1, 1))
        s = f + g
        assert _tech(s) == "Chebtech2"
        xs = jnp.linspace(-0.9, 0.9, 30)
        expected = jnp.sin(np.pi * xs) + xs
        assert float(jnp.max(jnp.abs(s(xs) - expected))) < 1e-12
