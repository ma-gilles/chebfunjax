"""Independent direct helper controls for complex PCG/MINRES dtype support.

Provenance
----------
MATLAB sources: @chebop/pcg.m and @chebop/minres.m;
Chebfun commit 7574c77. Source BC validation uses isa(value, 'double');
complex values are accepted and logical values are rejected. These focused
checks complement, but do not replace, the independent manufactured-solution
controls in complex_pcg_minres_source_controls_p67_draft_20261005.py.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators import krylov
from chebfunjax.operators.krylov import _ip, _minres_basic_correction


@pytest.mark.parametrize(
    "value",
    [
        2.0,
        2 + 3j,
        jnp.asarray(2.0),
        jnp.asarray(2.0 + 3.0j),
    ],
    ids=["python-real", "python-complex", "jax-real-0d", "jax-complex-0d"],
)
def test_double_scalar_adapter_accepts_numeric_scalars(value):
    assert krylov._is_double_scalar(value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        jnp.asarray(True),
        jnp.asarray([2.0]),
        jnp.asarray([2.0 + 3.0j]),
        jnp.asarray(2, dtype=jnp.int32),
    ],
    ids=["python-bool", "jax-bool-0d", "jax-real-vector", "jax-complex-vector", "jax-integer-0d"],
)
def test_double_scalar_adapter_rejects_logical_arrays_and_nonscalars(value):
    assert not krylov._is_double_scalar(value)


class _VectorForInner:
    def __init__(self, values):
        self.values = jnp.asarray(values)

    def inner(self, other):
        return jnp.vdot(self.values, other.values)


def test_ip_preserves_complex_cross_inner_and_real_self_inner():
    u = _VectorForInner([1.0 + 2.0j, -0.5 + 0.25j])
    v = _VectorForInner([0.25 - 1.0j, 2.0 + 0.5j])

    cross_expected = jnp.vdot(u.values, v.values)
    self_expected = jnp.vdot(u.values, u.values)
    cross_actual = _ip(u, v)
    self_actual = _ip(u, u)

    assert jnp.issubdtype(cross_actual.dtype, jnp.complexfloating)
    assert jnp.allclose(cross_actual, cross_expected, rtol=0.0, atol=0.0)
    assert jnp.allclose(self_actual, self_expected, rtol=0.0, atol=0.0)
    assert float(jnp.imag(self_actual)) == 0.0


def test_minres_qr_correction_retains_complex_rhs_with_real_matrix():
    # An independent exact-span case: rhs=A@expected, so the basic QR solution
    # is uniquely determined without comparing to a least-squares routine.
    matrix = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=jnp.float64,
    )
    expected = jnp.asarray([1.0 + 2.0j, -0.5 + 1.0j], dtype=jnp.complex128)
    rhs = matrix @ expected

    actual = _minres_basic_correction(matrix, rhs)

    assert jnp.issubdtype(actual.dtype, jnp.complexfloating)
    assert jnp.allclose(actual, expected, rtol=100 * np.finfo(float).eps, atol=0.0)
    assert jnp.allclose(matrix @ actual, rhs, rtol=100 * np.finfo(float).eps, atol=0.0)
