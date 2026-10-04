"""Port of MATLAB Chebfun tests/trigtech/test_isempty.m.

The source predicate checks object cardinality and the stored values field;
the no-argument constructor and empty concatenation are tested directly.

Provenance
----------
MATLAB source : tests/trigtech/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech


def _tt(f):
    return Trigtech.from_function(f)


class TestTrigtechIsempty:
    def test_empty(self):
        f = Trigtech()
        assert f.isempty()

    def test_scalar_nonempty(self):
        f = _tt(lambda x: jnp.sin(200 * jnp.pi * x))
        assert not f.isempty()

    def test_array_nonempty(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(200 * jnp.pi * x), jnp.cos(200 * jnp.pi * x)], axis=-1))
        assert not f.isempty()

    def test_concatenated_nonempty(self):
        f = Trigtech.horzcat(_tt(lambda x: jnp.sin(200 * jnp.pi * x)),
                             _tt(lambda x: jnp.sin(200 * jnp.pi * x)))
        assert not f.isempty()

    def test_concatenated_empty(self):
        f = Trigtech.horzcat(Trigtech(), Trigtech())
        assert f.isempty()
