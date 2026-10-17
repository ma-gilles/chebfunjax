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
