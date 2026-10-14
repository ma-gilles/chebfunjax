"""Port of MATLAB Chebfun tests/spherefunv/test_empty.m (Fable 5).

FIXED: Spherefunv.empty()/isempty() added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/spherefunv/test_empty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv


class TestSpherefunvEmpty:
    def test_empty_and_nonempty(self):
        assert Spherefunv.empty().isempty()
        v = Spherefunv(Spherefun.from_function(lambda lam, th: jnp.cos(th)), Spherefun.from_function(lambda lam, th: jnp.sin(th) * jnp.cos(lam)))
        assert not v.isempty()

    def test_native_empty_object_operations_do_not_raise(self):
        # Literal operation set inside the single native try/catch predicate.
        F = Spherefunv.empty()
        evaluation = F(1, 1)
        assert jnp.asarray(evaluation).shape == (0, 0)  # supplemental source shape
        operations = (
            lambda: F.cross(F),
            lambda: F.transpose(),
            lambda: F.curl(),
            lambda: F.divergence(),
            lambda: F.dot(F),
            lambda: F.isempty(),
            lambda: F - F,
            lambda: F.times(F),
            lambda: F.norm(),
            lambda: F + F,
            lambda: F.power(2),
            lambda: F @ F,
            lambda: F.ctranspose(),
            lambda: F.vorticity(),
            lambda: F.normal(),
        )
        for operation in operations:
            operation()
        assert True  # original MATLAB pass: none of the operations raised

    def test_jit_keeps_empty_and_row_static_state(self):
        evaluate = eqx.filter_jit(
            lambda field, lam, theta: field(lam, theta))
        empty_values = evaluate(Spherefunv.empty(), jnp.asarray(0.2),
                                jnp.asarray(0.7))
        assert jnp.asarray(empty_values).shape == (0, 0)

        component = Spherefun.from_function(lambda lam, theta: jnp.cos(theta))
        column = Spherefunv(component, component, component)
        row = column.transpose()
        row_size = evaluate_orientation(row)
        assert row_size == (float('inf'), float('inf'), 3)
        values = evaluate(column, jnp.asarray(0.2), jnp.asarray(0.7))
        assert len(values) == 3
        assert all(jnp.isfinite(value) for value in values)


def evaluate_orientation(field):
    # Applying an outer filter_jit forces Equinox to flatten/unflatten fields.
    return eqx.filter_jit(lambda value: value.size)(field)
