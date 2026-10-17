"""Literal original predicates 1--6 for MATLAB test_subspace.m.

This supplements test_subspace_matlab.py, whose single-interval translation
predates Quasimatrix support for different interior breakpoints.

Provenance
----------
MATLAB source : tests/chebfun/test_subspace.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain
from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(np.finfo(np.float64).eps)
PI = np.pi
SQPI = np.sqrt(PI)
A_DOMAIN = (0.0, 1.0, 2 * PI)
F_DOMAIN = (0.0, 2.0, 2 * PI)


def _original_operands():
    # MATLAB A is one array-valued Chebfun with breakpoints [0, 1, 2*pi].
    A = cj.chebfun(
        lambda t: jnp.stack(
            (jnp.ones_like(t) / jnp.sqrt(2.0), jnp.cos(t),
             jnp.sin(2 * t), jnp.sin(3 * t)), axis=-1
        ) / SQPI,
        domain=A_DOMAIN,
    )
    # MATLAB f has a distinct interior breakpoint at 2.
    f = cj.chebfun(lambda t: jnp.sin(10 * t) / SQPI, domain=F_DOMAIN)
    cols = [A[:, i] for i in range(4)]
    return A, cols, f


def _vscale(f):
    return float(f.vscale)


def test_original_subspace_predicates_1_to_5():
    A, cols, f = _original_operands()
    alpha = (1e-10, PI / 5, PI / 2 - 1e-10)

    # Original predicates 1--3: both argument orders, unchanged 1e3*eps bound.
    for k, angle in enumerate(alpha, start=1):
        B = np.cos(angle) * cols[k - 1] + np.sin(angle) * f
        assert abs(float(cj.subspace(cols, [B])) - angle) < 1e3 * EPS
        assert abs(float(cj.subspace([B], cols)) - angle) < 1e3 * EPS

    # Original predicate 4: A(:,1:2) versus A(:,3:4), unchanged bound.
    assert abs(float(cj.subspace(cols[0:2], cols[2:4])) - PI / 2) \
        < 10 * _vscale(A) * EPS

    # Original predicate 5: A(:,1:3) versus A(:,4), unchanged bound.
    assert abs(float(cj.subspace(cols[0:3], cols[3:4])) - PI / 2) \
        < 10 * _vscale(A) * EPS


def test_original_subspace_predicate_6_row_chebfuns():
    A, cols, _ = _original_operands()
    # MATLAB At=A.'; the list adapter transposes each original column and
    # subspace() maps the row spans back to their corresponding columns.
    At = [column.transpose() for column in cols]
    assert abs(float(cj.subspace(At[0:2], At[2:4])) - PI / 2) \
        < 10 * _vscale(A) * EPS


def test_native_array_and_quasimatrix_inputs_match_column_adapter():
    A, cols, f = _original_operands()
    B = jnp.cos(PI / 5) * cols[1] + jnp.sin(PI / 5) * f
    expected = float(cj.subspace(cols, [B]))
    assert abs(float(cj.subspace(A, B)) - expected) < 1e3 * EPS
    assert abs(float(cj.subspace(Quasimatrix(cols, A.domain), [B]))
               - expected) < 1e3 * EPS


def test_complex_small_angle_and_operator_residual():
    f = cj.chebfun(lambda x: jnp.exp(0.5j * PI * x), domain=(-1.0, 1.0))
    g = cj.chebfun(lambda x: jnp.exp(-0.5j * PI * x), domain=(-1.0, 1.0))
    # These complex columns are L2-orthogonal; the small-angle case exercises
    # the native sine recomputation after the complex QR/SVD path.
    assert abs(float(cj.subspace([f], [g])) - PI / 2) < 1e3 * EPS
    angle = 1e-10
    h = jnp.cos(angle) * f + jnp.sin(angle) * g
    assert abs(float(cj.subspace([f], [h])) - angle) < 1e3 * EPS


def test_native_transpose_and_domain_error_branches():
    f = cj.chebfun(lambda x: 1.0 + 0.0 * x, domain=(-1.0, 1.0))
    g = cj.chebfun(lambda x: x, domain=(-1.0, 1.0))
    row = f.T
    try:
        cj.subspace(row, f)
    except ValueError as exc:
        assert "Dimension mismatch (transpose)" in str(exc)
    else:
        raise AssertionError("row/column orientation mismatch was accepted")
    other = cj.chebfun(lambda x: x, domain=(0.0, 1.0))
    for left, right in (([f], [other]), (row, other)):
        try:
            cj.subspace(left, right)
        except ValueError as exc:
            assert "Domain mismatch" in str(exc)
        else:
            raise AssertionError("domain mismatch was accepted")
    assert abs(float(cj.subspace([row], [g.T])) - PI / 2) < 1e3 * EPS


def test_native_orth_rank_cutoff_uses_source_length_and_scale():
    domain = Domain((-1.0, 1.0))
    # The pinned MATLAB probe records length(A)=13 for two columns. At the
    # original scale, 16*eps is below `length(A)*eps(max(s))` while 128*eps
    # is above it; native orth() returns one versus two columns, respectively.
    f_coeffs = jnp.zeros((13,), dtype=jnp.float64).at[0].set(1.0)
    f = Chebfun.from_coeffs(f_coeffs, domain)
    x_coeffs = jnp.zeros((13,), dtype=jnp.float64).at[1].set(1.0)
    x = Chebfun.from_coeffs(x_coeffs, domain)
    below_coeffs = f_coeffs.at[12].set(16.0 * EPS)
    above_coeffs = f_coeffs.at[12].set(128.0 * EPS)
    below = Chebfun.from_coeffs(below_coeffs, domain)
    above = Chebfun.from_coeffs(above_coeffs, domain)
    # T12 is even and x is odd. A dropped perturbation leaves span(f), which
    # is contained in span(f,x); retaining it makes the second directions
    # orthogonal. The native probe returns near-zero and 1.5651038994694781; the
    # pi/4 split tests the rank decision without claiming exact-angle accuracy
    # for this ill-conditioned cutoff case.
    below_angle = float(cj.subspace([f, below], [f, x]))
    above_angle = float(cj.subspace([f, above], [f, x]))
    assert below_angle < PI / 4
    assert above_angle > PI / 4


def test_native_single_column_qr_shortcut_and_noncollatable_fallback():
    domain = Domain((-1.0, 1.0))
    left_sing = Singfun(Chebtech2.from_coeffs(jnp.asarray([1.0])), (0.5, 0.0))
    right_sing = Singfun(Chebtech2.from_coeffs(jnp.asarray([1.0])), (0.0, 0.5))
    f = Chebfun(funs=[_Piece(tech=left_sing, interval=(-1.0, 1.0))],
                domain=domain)
    g = Chebfun(funs=[_Piece(tech=right_sing, interval=(-1.0, 1.0))],
                domain=domain)
    # Native qr() normalizes a single column before quasi2cheb, even for a
    # noncollatable FUN; the two-column case reaches the abstractQR fallback.
    assert abs(float(cj.subspace([f], [f]))) < 1e-12
    assert abs(float(cj.subspace([f, g], [f + g]))) < 1e-10



def test_native_qr_rejects_unbounded_domain_before_single_column_shortcut():
    domain = Domain((-jnp.inf, jnp.inf))
    f = Chebfun.from_coeffs(jnp.asarray([1.0]), domain)
    try:
        cj.subspace([f], [f])
    except ValueError as exc:
        assert "CHEBFUN:CHEBFUN:qr:infdomain" in str(exc)
    else:
        raise AssertionError("unbounded-domain QR was accepted")
