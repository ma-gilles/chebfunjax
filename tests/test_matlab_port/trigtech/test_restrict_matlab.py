"""Port of MATLAB Chebfun tests/trigtech/test_restrict.m (Fable 5).

Provenance
----------
MATLAB source : tests/trigtech/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


class TestTrigtechRestrict:
    def test_empty(self):
        # pass(1): restrict of an empty trigtech is empty
        assert Trigtech.empty().restrict(-0.5, 0.5).isempty()

    def test_full_domain_is_identity_and_bad_interval_identifiers(self):
        f = Trigtech.from_function(lambda x: jnp.sin(2 * jnp.pi * x))
        assert f.restrict(-1.0, 1.0) is f
        cases = [
            ([-1.0, 3.0], "TRIGTECH:restrict:badinterval"),
            ([-2.0, 1.0], "TRIGTECH:restrict:badinterval"),
            ([-1.0, -0.25, 0.3, 0.1, 1.0],
             "TRIGTECH:restrict:badinterval"),
        ]
        for interval, identifier in cases:
            with pytest.raises(ValueError, match=identifier):
                f.restrict(interval)

    def test_original_native_spotchecks(self):
        cases = [
            (lambda x: jnp.sin(2 * jnp.pi * x) - 1.0, (-0.5, 0.5)),
            (lambda x: 3.0 / (5.0 - 4.0 * jnp.cos(2 * jnp.pi * x)),
             (-0.5, 0.5)),
            (lambda x: jnp.cos(4 * jnp.pi * x), (-0.25, 0.25)),
        ]
        for fun, (a, b) in cases:
            f = Trigtech.from_function(fun)
            restricted = f.restrict(a, b)
            x = jnp.asarray(np.linspace(a, b, 100))
            mapped = (2.0 / (b - a)) * (x - a) - 1.0
            error = jnp.max(jnp.abs(fun(x) - restricted(mapped)))
            assert error < 100 * restricted.vscale * EPS

    def test_native_overwritten_pass_nine_predicate(self):
        f = Trigtech.from_function(lambda x: jnp.sin(4 * jnp.pi * x))
        h1 = f.restrict(-1.0, -0.5)
        h2 = f.restrict(0.0, 0.5)
        expected = Trigtech.from_function(lambda x: -jnp.sin(jnp.pi * x))
        x = jnp.asarray(np.linspace(-1.0, 1.0, 100))
        error = (jnp.max(jnp.abs(expected(x) - h1(x)))
                 + jnp.max(jnp.abs(expected(x) - h2(x))))
        assert error < 10 * EPS

    def test_array_valued(self):
        # pass(10): array-valued restriction of [sin(2pi x) cos(4pi x)
        # exp(cos(2pi x))] to [-0.5, 0.5].  MATLAB's pass(10) is commented out
        # in the reference, but the operation is well-defined, so we port its
        # (disabled) spec here at the same tolerance the scalar spot-check uses.
        # Supplemental control for the commented native array-valued case.
        a, b = -0.5, 0.5

        def fun(x):
            return jnp.stack(
                [jnp.sin(2 * jnp.pi * x), jnp.cos(4 * jnp.pi * x), jnp.exp(jnp.cos(2 * jnp.pi * x))],
                axis=-1,
            )

        f = Trigtech.from_function(fun)
        g = f.restrict(a, b)
        ts = jnp.asarray(np.linspace(a, b, 100))
        mapx = (2.0 / (b - a)) * (ts - a) - 1.0
        err = float(jnp.max(jnp.abs(fun(ts) - g(mapx))))
        assert err < 100 * g.vscale * EPS
