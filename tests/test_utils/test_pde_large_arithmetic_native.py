"""Native CPU source arithmetic for polynomial PDE endpoint/DAE paths.

Provenance: Chebfun pdeSolve.m pin7574c77 and installed MATLAB R2025b
sum, cos and dense matrix-vector products. Saved operands remain exact.
"""

import hashlib
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.chebfun1d._pde_ndf.linear_boundary import _endpoint_sum
from chebfunjax.chebfun1d._pde_ndf.source_arithmetic import matvec
from chebfunjax.utils._fdlibm_boundary_cosine import _cos_boundary_angles

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu" or not jax.config.x64_enabled,
    reason="Captured MATLAB arithmetic contract is CPU binary64",
)

FIXTURE = (
    Path(__file__).parents[1]
    / "references"
    / "pde_large_arithmetic_native"
    / "native_arithmetic_corpus.mat"
)
FIXTURE_SHA256 = "f8e76744ae115be842eaef51ad504e559fca32fca88365abe39965b08be34476"


def _unwrap(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value


def _bits(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@lru_cache(maxsize=1)
def _native():
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    return loadmat(
        FIXTURE,
        mat_dtype=True,
        squeeze_me=False,
        struct_as_record=False,
    )


def _records(name):
    return [_unwrap(item) for item in _native()[name].flat]


def test_all_captured_dense_and_source_matvec_actions_match_native_bits():
    """15 shapes x 2 dense variants, plus 5 actual/source-formula actions."""
    corpus = _native()
    comparisons = 0
    for case in _records("gemvCases"):
        for variant in ("signed", "mixed"):
            matrix = jnp.asarray(getattr(case, f"{variant}_A"))
            vector = jnp.asarray(getattr(case, f"{variant}_x")).reshape(-1)
            expected = getattr(case, f"{variant}_y")
            assert expected.shape == (matrix.shape[0], 1)
            actual = matvec(matrix, vector)
            assert actual.shape == (matrix.shape[0],)
            _bits(actual, expected)
            comparisons += 1

    ops = _unwrap(corpus["nativeOperators"])
    for matrix_name, vector_name, result_name in (
        ("B", "initial", "B_initial"),
        ("B", "probe", "B_probe"),
        ("P", "initial", "P_initial"),
        ("P", "probe", "P_probe"),
    ):
        _bits(
            matvec(
                jnp.asarray(getattr(ops, matrix_name)),
                jnp.asarray(getattr(ops, vector_name)).reshape(-1),
            ),
            getattr(ops, result_name),
        )
        comparisons += 1

    supplemental = _unwrap(corpus["constructorFormula"])
    assert bool(_unwrap(supplemental.available))
    _bits(
        matvec(jnp.asarray(supplemental.B), jnp.asarray(supplemental.x).reshape(-1)),
        supplemental.y,
    )
    comparisons += 1
    assert comparisons == 35


def test_private_endpoint_sum_matches_all_2054_native_source_vectors():
    """Both general families at n=1..514 and Neumann pre-endpoint rows n>=2."""
    comparisons = 0
    sum_row = jax.jit(_endpoint_sum)
    for case in _records("sumCases"):
        n = int(np.asarray(case.n).item())
        for vector_field, result_field in (
            ("mixed_scale", "mixed_sum"),
            ("alternating_rational", "alternating_sum"),
        ):
            _bits(
                sum_row(jnp.asarray(getattr(case, vector_field)).reshape(-1)),
                getattr(case, result_field),
            )
            comparisons += 1
        if bool(np.asarray(case.neumann_applicable).item()):
            for vector_field, result_field in (
                ("left_signed_preendpoint", "left_preendpoint_sum"),
                ("right_signed_preendpoint", "right_preendpoint_sum"),
            ):
                row = np.asarray(getattr(case, vector_field)).reshape(-1)
                assert row.size == n
                _bits(sum_row(jnp.asarray(row)), getattr(case, result_field))
                comparisons += 1
        else:
            assert n == 1
    assert comparisons == 2054


def test_bounded_cosine_matches_saved_native_binary64_bits():
    """Compare the dense, endpoint, all-Neumann-angle, and branch corpus."""
    corpus = _native()
    angle_parts = [np.asarray(corpus["denseAngles"]).reshape(-1)]
    cosine_parts = [np.asarray(corpus["denseCos"]).reshape(-1)]
    for angles, expected in zip(corpus["endpointAngles"].flat, corpus["endpointCos"].flat):
        angle_parts.append(np.asarray(angles).reshape(-1))
        cosine_parts.append(np.asarray(expected).reshape(-1))
    for case in _records("sumCases"):
        angle_parts.append(np.asarray(case.theta).reshape(-1))
        cosine_parts.append(np.asarray(case.cos_theta).reshape(-1))
    angle_parts.append(np.asarray(corpus["branchAngles"]).reshape(-1))
    cosine_parts.append(np.asarray(corpus["branchCos"]).reshape(-1))
    angles = np.concatenate(angle_parts)
    expected = np.concatenate(cosine_parts)
    assert angles.shape == expected.shape
    assert np.all(np.isfinite(angles))
    in_domain = np.abs(angles) <= np.pi
    angles, expected = angles[in_domain], expected[in_domain]
    assert angles.size == 139783
    assert np.all(np.isfinite(expected))
    _bits(_cos_boundary_angles(jnp.asarray(angles)), expected)


def test_bounded_cosine_outside_domain_returns_nan():
    outside = np.asarray(
        [np.nextafter(np.pi, np.inf), np.nextafter(-np.pi, -np.inf), np.inf, -np.inf, np.nan],
        dtype=np.float64,
    )
    actual = np.asarray(_cos_boundary_angles(jnp.asarray(outside)))
    assert np.all(np.isnan(actual))


def test_bounded_cosine_jvp_and_vjp_follow_cosine_derivative():
    angles = jnp.asarray([0.125, 0.9, 2.25], dtype=jnp.float64)
    tangent = jnp.asarray([0.5, -1.25, 0.75], dtype=jnp.float64)
    _, tangent_out = jax.jvp(_cos_boundary_angles, (angles,), (tangent,))
    np.testing.assert_allclose(
        tangent_out,
        -jnp.sin(angles) * tangent,
        rtol=2.0e-14,
        atol=2.0e-15,
    )
    weights = jnp.asarray([-0.75, 0.25, 1.5], dtype=jnp.float64)
    _, pullback = jax.vjp(_cos_boundary_angles, angles)
    (cotangent_in,) = pullback(weights)
    np.testing.assert_allclose(
        cotangent_in,
        -jnp.sin(angles) * weights,
        rtol=2.0e-14,
        atol=2.0e-15,
    )


def test_extended_matvec_jvp_and_vjp_linear_identities():
    """Use exact sparse data on a qualified-size shape for derivative controls."""
    rows = jnp.arange(6)
    cols = (17 * rows + 5) % 303
    matrix = (
        jnp.zeros((6, 303), dtype=jnp.float64)
        .at[rows, cols]
        .set(jnp.asarray([1.0, -2.0, 4.0, -8.0, 16.0, -32.0]))
    )
    vector = jnp.arange(303, dtype=jnp.float64) % 7 - 3
    dmatrix = (
        jnp.zeros_like(matrix).at[rows, cols].set(jnp.asarray([2.0, 1.0, -1.0, 2.0, -2.0, 1.0]))
    )
    dvector = jnp.arange(303, dtype=jnp.float64) % 5 - 2
    _, tangent_out = jax.jvp(
        matvec,
        (matrix, vector),
        (dmatrix, dvector),
    )
    expected_tangent = dmatrix[rows, cols] * vector[cols] + matrix[rows, cols] * dvector[cols]
    np.testing.assert_array_equal(tangent_out, expected_tangent)

    cotangent = jnp.asarray([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    _, pullback = jax.vjp(matvec, matrix, vector)
    matrix_bar, vector_bar = pullback(cotangent)
    expected_matrix_bar = cotangent[:, None] * vector[None, :]
    expected_vector_bar = jnp.zeros_like(vector).at[cols].set(matrix[rows, cols] * cotangent)
    np.testing.assert_array_equal(matrix_bar, expected_matrix_bar)
    np.testing.assert_array_equal(vector_bar, expected_vector_bar)


def test_matvec_generic_dtype_shape_and_nonfinite_fallbacks():
    for dtype in (jnp.float32, jnp.complex128):
        matrix = jnp.arange(12, dtype=jnp.float64).reshape(3, 4).astype(dtype) / 7
        vector = jnp.asarray([1, -2, 3, -4], dtype=dtype)
        np.testing.assert_array_equal(matvec(matrix, vector), matrix @ vector)
    for shape in ((1, 33), (515, 2), (2, 515)):
        matrix = jnp.arange(shape[0] * shape[1], dtype=jnp.float64).reshape(shape) / 13
        vector = jnp.linspace(-1.0, 1.0, shape[1])
        np.testing.assert_array_equal(matvec(matrix, vector), matrix @ vector)
    matrix = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    vector = jnp.asarray([1.0, -2.0, 3.0])
    nonfinite = matrix.at[0, 1].set(jnp.inf)
    np.testing.assert_array_equal(matvec(nonfinite, vector), nonfinite @ vector)


def test_endpoint_sum_outside_qualified_size_preserves_serial_policy():
    # Preserve the previous helper policy outside captured lengths1..514;
    # this is a compatibility check, not a native515-sum parity claim.
    source = np.asarray(_records("sumCases")[-1].mixed_scale).reshape(-1)
    row = np.concatenate((source, np.asarray([1.0 / 3.0])))
    assert row.shape == (515,)
    expected = np.float64(0.0)
    for value in row:
        expected = expected + value
    _bits(_endpoint_sum(jnp.asarray(row)), expected)
