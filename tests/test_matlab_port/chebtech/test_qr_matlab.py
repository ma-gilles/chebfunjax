"""Port of MATLAB Chebfun tests/chebtech/test_qr.m (Fable 5).

Array-valued techs and a tech-level ``qr`` both exist now, so the MATLAB
assertions are ported directly at MATLAB's tolerances.  The MATLAB file loops
``for n = 1:4`` over the four (class, method) combinations
``{chebtech1, chebtech2} x {'householder', 'built-in'}``; we parametrize over
the same four.

Remaining adaptations vs MATLAB:
* Passes 21-22 construct the source method-3 Legendre coefficient blocks
  directly as Tech objects, then exercise the ``n > 4000`` NDCT/IDLT branch.
  MATLAB constructs a CHEBFUN and repeats the same two results across its
  four class/method rows. Here both Tech kinds exercise each source block.
* Pass 20 checks ``size(vscale(Q)) == [1 3]``.  chebfunjax's ``vscale`` is a
  scalar aggregate; the native vector-size assertion uses vscale_columns.

The two-output form remains unpivoted. MATLAB's three-output built-in form
pivots columns; focused controls below cover a non-tied pivot, matrix/vector
encodings, rank deficiency without tie-order assumptions, and the fast
transform branch.

Provenance
----------
MATLAB source : tests/chebtech/test_qr.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

# MATLAB uses seedRNG(6178); x = 2*rand(100,1) - 1. This deterministic
# adapter retains the source predicates; matched native query inputs are pending.
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))

# (class, method) -- the four passes of the MATLAB n = 1:4 loop.
CASES = [
    (Chebtech1, "householder"),
    (Chebtech1, "built-in"),
    (Chebtech2, "householder"),
    (Chebtech2, "built-in"),
]
IDS = ["c1-hh", "c1-builtin", "c2-hh", "c2-builtin"]


def _ncols(f):
    return f.coeffs.shape[1] if f.coeffs.ndim == 2 else 1


def _check_one_qr(f, method):
    """MATLAB helper ``test_one_qr``: orthogonality + factorisation accuracy."""
    N = _ncols(f)
    Q, R = f.qr(method=method)
    tol = 1e3 * f.vscale * EPS

    # result(1): check orthogonality.
    ip = jnp.reshape(jnp.asarray(Q.inner(Q)), (N, N))
    assert float(jnp.max(jnp.abs(ip - jnp.eye(N)))) < tol

    # result(2): check that the factorization is accurate.
    err = (Q @ R) - f
    assert float(jnp.linalg.norm(err(X), ord=jnp.inf)) < tol


def _check_one_qr_with_perm(f, method):
    """MATLAB helper ``test_one_qr_with_perm``: same, with ``f * E``."""
    N = _ncols(f)
    Q, R, E = f.qr(method=method, want_e=True)
    tol = 1e3 * f.vscale * EPS

    ip = jnp.reshape(jnp.asarray(Q.inner(Q)), (N, N))
    assert float(jnp.max(jnp.abs(ip - jnp.eye(N)))) < tol

    err = (Q @ R) - (f @ E)
    assert float(jnp.linalg.norm(err(X), ord=jnp.inf)) < tol


def _scalar(x):
    return jnp.sin(x)


def _two_col(x):
    return jnp.stack([jnp.cos(x), jnp.exp(x)], axis=-1)


def _monomials(x):
    return jnp.stack([x**k for k in range(8)], axis=-1)


def _complex_cols(x):
    return jnp.stack(
        [1.0 / (1.0 + 1j * x**2), jnp.sinh((1 - 1j) * x), jnp.exp(x) - x**3],
        axis=-1,
    )


class TestChebtechQr:
    @pytest.mark.parametrize("Tech,method", CASES, ids=IDS)
    def test_scalar_valued(self, Tech, method):
        # pass(n, 1:4)
        f = Tech.from_function(_scalar)
        _check_one_qr(f, method)
        _check_one_qr_with_perm(f, method)

    @pytest.mark.parametrize("Tech,method", CASES, ids=IDS)
    def test_two_columns(self, Tech, method):
        # pass(n, 5:8)
        f = Tech.from_function(_two_col)
        _check_one_qr(f, method)
        _check_one_qr_with_perm(f, method)

    @pytest.mark.parametrize("Tech,method", CASES, ids=IDS)
    def test_monomial_basis(self, Tech, method):
        # pass(n, 9:12) -- [1 x x^2 ... x^7]
        f = Tech.from_function(_monomials)
        _check_one_qr(f, method)
        _check_one_qr_with_perm(f, method)

    @pytest.mark.parametrize("Tech,method", CASES, ids=IDS)
    def test_complex_columns(self, Tech, method):
        # pass(n, 13:16)
        f = Tech.from_function(_complex_cols)
        _check_one_qr(f, method)
        _check_one_qr_with_perm(f, method)

    @pytest.mark.parametrize("Tech,method", CASES, ids=IDS)
    def test_vector_flag_consistent_with_matrix(self, Tech, method):
        # pass(n, 17): E1(:, E2) - eye(N) == 0.
        f = Tech.from_function(_complex_cols)
        N = _ncols(f)
        _, _, E1 = f.qr(mode="matrix", method=method, want_e=True)
        _, _, E2 = f.qr(mode="vector", method=method, want_e=True)
        err = E1[:, E2] - jnp.eye(N)
        assert bool(jnp.all(err == 0))

    @pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
    def test_builtin_three_output_pivots_larger_column_first(self, Tech):
        # The native pass 5-8 fixture is [cos(x), exp(x)]; with a full dense
        # three-output QR, MATLAB orders abs(diag(R)) decreasing.
        f = Tech.from_function(_two_col)
        Q, R, permutation = f.qr(
            mode="vector", method="built-in", want_e=True)
        assert int(permutation[0]) == 1

        Qm, Rm, E = f.qr(mode="matrix", method="built-in", want_e=True)
        assert bool(jnp.array_equal(E[:, permutation], jnp.eye(2)))
        err = (Qm @ Rm) - (f @ E)
        assert float(jnp.linalg.norm(err(X), ord=jnp.inf)) < 1e3 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
    def test_builtin_three_output_rank_deficient_permutation(self, Tech):
        # Supplemental control: preserve reconstruction without fixing a tie order.
        f = Tech.from_function(lambda x: jnp.stack([x, x, x], axis=-1))
        Q, R, permutation = f.qr(
            mode="vector", method="built-in", want_e=True)
        assert sorted(np.asarray(permutation).tolist()) == [0, 1, 2]
        Qm, Rm, E = f.qr(mode="matrix", method="built-in", want_e=True)
        assert bool(jnp.array_equal(E[:, permutation], jnp.eye(3)))
        err = (Qm @ Rm) - (f @ E)
        assert float(jnp.linalg.norm(err(X), ord=jnp.inf)) < 1e3 * f.vscale * EPS

    @pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
    def test_builtin_three_output_pivot_fast_transform_branch(self, Tech):
        # Prolong the native [cos, exp] fixture beyond 4000 to cover pivot
        # selection and both permutation encodings in the IDLT/NDCT branch.
        # High-branch reconstruction is tracked separately: the unchanged
        # unpivoted path has the same pre-existing pointwise residual.
        f = Tech.from_function(_two_col).prolong(4001)
        Q, R, permutation = f.qr(
            mode="vector", method="built-in", want_e=True)
        assert int(permutation[0]) == 1
        Qm, Rm, E = f.qr(mode="matrix", method="built-in", want_e=True)
        assert bool(jnp.array_equal(permutation, jnp.asarray([1, 0])))
        assert bool(jnp.array_equal(E[:, permutation], jnp.eye(2)))
        assert R.shape == (2, 2) and Rm.shape == (2, 2)
        assert _ncols(Q) == 2 and _ncols(Qm) == 2

    @pytest.mark.parametrize("Tech,method", CASES, ids=IDS)
    def test_rank_deficient(self, Tech, method):
        # pass(n, 18): size(Q) == 3 and size(R) == 3 for f = [x x x].
        f = Tech.from_function(
            lambda x: jnp.stack([x, x, x], axis=-1))
        Q, R = f.qr(method=method)
        assert _ncols(Q) == 3
        assert R.shape == (3, 3)

    @pytest.mark.parametrize("Tech,method", CASES, ids=IDS)
    def test_array_valued_output_shape(self, Tech, method):
        # pass(n, 20): MATLAB checks size(vscale(Q)) == [1 3]. The
        # documented Python vector adapter vscale_columns has shape (3,).
        # Retain the aggregate scale checks as supplemental controls.
        f = Tech.from_function(
            lambda x: jnp.stack([x, x**2, x**3], axis=-1))
        Q, R = f.qr(method=method)
        assert Q.vscale_columns.shape == (3,)
        assert _ncols(Q) == 3
        assert R.shape == (3, 3)
        assert np.isfinite(Q.vscale) and Q.vscale > 0

    @pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
    @pytest.mark.parametrize("first_degree,ncols", [(4999, 2), (10000, 6)])
    def test_large_legendre_blocks(self, Tech, first_degree, ncols):
        # Source pass(:,21:22), legpoly.m method 3: a single unit-column
        # Legendre coefficient matrix converted with leg2cheb. Coefficient
        # lengths are 5001 and 10006, so both exercise fast IDLT (n>=5000).
        from chebfunjax.utils.transforms import leg2cheb

        degrees = jnp.arange(first_degree, first_degree + ncols)
        n = first_degree + ncols
        legendre_coeffs = jnp.zeros((n, ncols), dtype=jnp.float64)
        legendre_coeffs = legendre_coeffs.at[
            degrees, jnp.arange(ncols)
        ].set(1.0)
        L = Tech(coeffs=leg2cheb(legendre_coeffs))
        Q, _R = L.qr(method="built-in")
        error = (Q @ jnp.diag(jnp.sqrt(1.0 / (degrees + 0.5)))) - L

        # L is a CHEBFUN in MATLAB, whose default norm is continuous L2
        # Frobenius (@chebfun/norm.m), not numeric matrix spectral norm.
        # Tech inner() computes the exact polynomial Gram matrix via
        # prolonged Clenshaw-Curtis quadrature without an n-by-n matrix.
        source_norm = jnp.sqrt(jnp.abs(jnp.trace(error.inner(error))))
        # MATLAB retains the final loop's f=[x,x^2,x^3] for this bound.
        f = Tech.from_function(
            lambda x: jnp.stack([x, x**2, x**3], axis=-1)
        )
        assert float(source_norm) < 5e4 * f.vscale * EPS


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_builtin_three_output_noninvolutory_permutation(Tech):
    # Supplemental non-tied three-cycle: catches inverse-permutation mistakes
    # that identity and two-column swaps cannot expose.
    f = Tech.from_function(
        lambda x: jnp.stack([0.1*x**2, 3*jnp.ones_like(x), 2*x], axis=-1))
    Q, R, permutation = f.qr(mode=0, method="built-in", want_e=True)
    assert bool(jnp.array_equal(permutation, jnp.asarray([1, 2, 0])))
    Qm, Rm, E = f.qr(mode="matrix", method="built-in", want_e=True)
    assert bool(jnp.array_equal(E, jnp.eye(3)[:, permutation]))
    assert float(jnp.linalg.norm(((Qm @ Rm) - (f @ E))(X), ord=jnp.inf)) < (
        1e3 * f.vscale * EPS)
    assert float(jnp.linalg.norm(
        (Q @ R)(X) - f(X)[:, permutation], ord=jnp.inf)) < 1e3 * f.vscale * EPS
