"""Diagnostic endpoint data-selection identity, not native numerical predicates."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.inverse import _restore_inverse_endpoints


@pytest.mark.parametrize("case", range(9))
def test_literal_endpoint_selection_bits(case):
    payload = np.array([0x7FF8000000000042], dtype=np.uint64).view(np.float64)[0]
    y = np.array(
        [0.0, 0.5, 1.0, np.nan, np.inf, -np.inf, np.nextafter(0.5, 1.0), np.nextafter(1.0, 0.0)]
    )
    result = np.array([-0.0, payload, np.inf, payload, -np.inf, np.inf, 7.0, 8.0])
    fa = 0.0
    fb = 1.0
    a = -2.0
    b = 3.0
    if case == 1:
        fa, fb = 1.0, 0.0
    if case == 2:
        y = np.array([0.0, -0.0, 1.0])
        result = np.array([np.nan, np.inf, payload])
        fa, fb = 0.0, -0.0
        a, b = -0.0, 0.0
    if case == 3:
        fa = np.nan
        fb = np.nan
    if case == 4:
        fa, fb = np.inf, -np.inf
    if case == 5:
        y = np.array(0.0)
        result = np.array(payload)
        a = -0.0
    if case == 6:
        y = y.reshape((2, 4))
        result = result.reshape((2, 4))
    if case == 7:
        y = np.empty(0)
        result = np.empty(0)
    if case == 8:
        fa = fb = 0.5
        a = -0.0
        b = np.inf
    # Literal independent selection: fb first, then fa overrides matching rows.
    expected = np.array(result, copy=True)
    expected[y == fb] = b
    expected[y == fa] = a
    actual = np.asarray(
        _restore_inverse_endpoints(jnp.asarray(y), jnp.asarray(result), fa, fb, a, b)
    )
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    eager = jnp.where(
        jnp.asarray(y) == fa, a, jnp.where(jnp.asarray(y) == fb, b, jnp.asarray(result))
    )
    assert actual.tobytes() == np.asarray(eager).tobytes()


def test_inherited_subnormal_jax_equality_is_not_changed():
    y = jnp.asarray([np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0), 0.0])
    result = jnp.asarray([7.0, 8.0, 9.0])
    fa = 0.0
    fb = 1.0
    a = -2.0
    b = 3.0
    eager = np.asarray(jnp.where(y == fa, a, jnp.where(y == fb, b, result)))
    fused = np.asarray(_restore_inverse_endpoints(y, result, fa, fb, a, b))
    assert fused.tobytes() == eager.tobytes()
    print(
        {
            "subnormal_original": eager.tolist(),
            "subnormal_fused": fused.tolist(),
            "numpy_exact_comparison": np.where(
                np.asarray(y) == fa, a, np.where(np.asarray(y) == fb, b, np.asarray(result))
            ).tolist(),
        },
        flush=True,
    )
