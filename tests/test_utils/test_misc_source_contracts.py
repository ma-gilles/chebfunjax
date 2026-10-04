"""Source-bound fixtures for scratch JAX-only utility candidates.

MATLAB provenance: Chebfun commit 7574c77680d7e82b79626300bf255498271a72df,
tests/misc/test_isSubset.m and matlab_harness/refs/misc.m abstractQR section.
The analytic QR identities are independent of the implementation.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.utils.misc import abstract_qr, isSubset


def test_abstract_qr_real_source_harness_shape_and_factorization():
    # A compact deterministic counterpart to misc.m's real 5x3 fixture.
    a = jnp.array([[1.0, 2.0, -1.0], [2.0, -1.0, 3.0],
                   [0.5, 1.0, 2.0], [-2.0, 0.25, 1.0],
                   [1.5, 3.0, -0.5]], dtype=jnp.float64)
    e = jnp.eye(5, 3, dtype=jnp.float64)
    q, r = abstract_qr(a, e, jnp.vdot)
    assert q.shape == (5, 3)
    assert r.shape == (3, 3)
    assert jnp.allclose(q.T @ q, jnp.eye(3), rtol=2e-13, atol=2e-13)
    assert jnp.allclose(q @ r, a, rtol=2e-13, atol=2e-13)
    assert jnp.allclose(jnp.tril(r, -1), 0.0, atol=0.0)


def test_abstract_qr_complex_uses_hermitian_projection():
    # The independent Gram identities distinguish vdot from a non-conjugating
    # dot product; E is the first two standard basis vectors in C^4.
    a = jnp.array([[1 + 2j, -1j], [2 - 1j, 3 + 0.5j],
                   [-0.25j, 1 - 2j], [1.5 + 0.25j, -2 + 1j]],
                  dtype=jnp.complex128)
    e = jnp.eye(4, 2, dtype=jnp.complex128)
    q, r = abstract_qr(a, e, jnp.vdot)
    eye = jnp.eye(2, dtype=jnp.complex128)
    assert jnp.iscomplexobj(q) and jnp.iscomplexobj(r)
    assert jnp.allclose(jnp.conj(q.T) @ q, eye, rtol=3e-13, atol=3e-13)
    assert jnp.allclose(q @ r, a, rtol=3e-13, atol=3e-13)
    assert jnp.allclose(jnp.tril(r, -1), 0.0, atol=0.0)


def test_abstract_qr_zero_column_uses_source_householder_fallback():
    a = jnp.array([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], dtype=jnp.float64)
    e = jnp.eye(3, 2, dtype=jnp.float64)
    q, r = abstract_qr(a, e, jnp.vdot, tol=1e-14)
    assert r[1, 1] == 0.0
    assert jnp.allclose(q.T @ q, jnp.eye(2), atol=2e-13)
    assert jnp.allclose(q @ r, a, atol=2e-13)


def test_is_subset_source_test_is_six_endpoint_domains_and_tolerance():
    tol = 100.0 * jnp.finfo(jnp.float64).eps
    assert isSubset([0, 2], [0, 2], tol)
    assert not isSubset([0, 2], [0, 1], tol)
    assert isSubset([-0.7, -0.5], [-1, 1], tol)
    assert isSubset([-jnp.finfo(jnp.float64).eps, 1], [0, 1], tol)
    assert isSubset([-0.7, -0.5, 1, 2], [-1, 1, -1, 3], tol)
    assert not isSubset([-0.7, -0.5, 1, 2], [-1, 1, -1, 1], tol)
    assert isSubset([-1, 1, 0, 1, -1, 1], [-1, 1, -1, 1, -1, 1], tol)
    assert not isSubset([-1, 1, -2, 1, -1, 1], [-1, 1, -1, 1, -1, 1], tol)
    assert not isSubset([-jnp.finfo(jnp.float64).eps, 1], [0, 1], 0.0)


def test_is_subset_empty_input_precedes_size_check_and_mismatch_errors():
    assert isSubset([], [0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="same number of entries"):
        isSubset([0.0, 1.0], [0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="endpoint pairs"):
        isSubset([0.0], [0.0])


def test_standard_chop_matlab_half_away_plateau_window():
    """Source j=6 samples position 13, delaying the plateau until j=12.

    With tol=.01 and a=tol**.8, the j=6 predicate threshold is .6.
    Position 12 is .8*a and position 13 is .5*a: ties-to-even accepts
    the earlier plateau and returns 5, while MATLAB's source returns 12.
    The source log10+bias minimum for the j=12 window of 20 is position 13.
    """
    from chebfunjax.utils.misc import standard_chop

    a = .01 ** .8
    coeffs = jnp.array([1.] * 5 + [a] * 6 + [.8*a] + [.5*a] * 8)
    assert standard_chop(coeffs, .01) == 12
    assert standard_chop(1j * coeffs, .01) == 12


@pytest.mark.matlab
@pytest.mark.parametrize("fixture", [1, 2])
def test_abstract_qr_matches_saved_matlab_factors(fixture):
    """Source phase choices determine both factors for the fixed basis E."""
    from tests.conftest import load_matlab_ref

    ref = load_matlab_ref("misc.mat")
    a = jnp.asarray(ref[f"aqr_A{fixture}"])
    e = jnp.asarray(ref[f"aqr_E{fixture}"])
    q, r = abstract_qr(a, e, jnp.vdot)
    assert jnp.allclose(q, jnp.asarray(ref[f"aqr_Q{fixture}"]),
                        rtol=1e-12, atol=1e-13)
    assert jnp.allclose(r, jnp.asarray(ref[f"aqr_R{fixture}"]),
                        rtol=1e-12, atol=1e-13)
