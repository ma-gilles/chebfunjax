# ruff: noqa: E731
"""Historical deterministic Singfun factory controls retained as supplemental.

These explicit-exponent comparisons are not the literal native auto/data/pref
routes. Those routes are restored in test_make_matlab.py. Assertion bodies
are preserved from the prior port.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.fun.singfun import Singfun

_REASON = "chebfunjax Singfun has no make() factory method"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunMakeLegacy:
    def test_handle_only(self):
        # Historical supplemental reference uses supplied exponents.
        a, b = 0.3, 0.4
        fh = lambda x: jnp.sin(x) / ((1 + x) ** a * (1 - x) ** b)
        f = _sf(fh, (-a, -b))
        assert f.make(fh) == f

    def test_handle_and_exponents(self):
        a, b = 0.3, 0.4
        fh = lambda x: jnp.sin(x) * (1 + x) ** a * (1 - x) ** b
        f = _sf(fh, (a, b))
        assert f.make(fh, (a, b)) == f

    def test_handle_and_singtype(self):
        # Historical supplemental comparison supplies integer exponents explicitly.
        a, b = 3, 4
        fh = lambda x: jnp.exp(x) / ((1 + x) ** a * (1 - x) ** b)
        f = _sf(fh, (-a, -b))
        assert f.make(fh, (-a, -b)) == f

    def test_handle_exponents_pref(self):
        a, b = 0.3, 0.4
        fh = lambda x: jnp.exp(jnp.sin(x)) / ((1 + x) ** a * (1 - x) ** b)
        f = _sf(fh, (-a, -b))
        assert f.make(fh, (-a, -b)) == f

    def test_handle_singtype_pref(self):
        a, b = 0.3, 0.4
        fh = lambda x: jnp.sin(jnp.exp(jnp.cos(x))) * (1 + x) ** a * (1 - x) ** b
        f = _sf(fh, (a, b))
        assert f.make(fh, (a, b)) == f

    def test_all_arguments(self):
        a, b = 2, 3
        fh = lambda x: jnp.exp(jnp.sin(x ** 2)) / ((1 + x) ** a * (1 - x) ** b)
        f = _sf(fh, (-a, -b))
        assert f.make(fh, (-a, -b)) == f
