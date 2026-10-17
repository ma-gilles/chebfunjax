"""Unchanged MATLAB test_legcoeffs.m predicates 1–6.

The original scalar-only checks remain supplemental in
``test_legcoeffs_matlab.py``. This module retains the native vector-valued,
piecewise, and row-Chebfun operands and exact reference matrices.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = np.finfo(np.float64).eps


def _max_abs(values):
    return float(jnp.max(jnp.abs(jnp.asarray(values))))


def test_original_pass3_array_valued_smooth():
    f = cj.chebfun(
        lambda x: jnp.stack([jnp.sin(x - 0.3), jnp.cos(x - 0.3)], axis=-1))
    actual = f.legcoeffs(5)
    expected = jnp.asarray([
        [-0.2486716793299505096289379, 0.8038879363274420091869815],
        [0.8631522851187122776801783, 0.2670042907205986623437124],
        [0.0916630569532407635738723, -0.2963217439563513424659842],
        [-0.0602302090841307493414730, -0.0186313869912884662151106],
        [-0.0026889804057628215030252, 0.0086927426357443276344859],
    ])
    assert actual.shape == (5, 2)
    assert _max_abs(actual - expected) < 10 * float(f.vscale) * EPS


def test_original_pass4_piecewise_scalar():
    f = cj.chebfun(lambda x: jnp.abs(x - 0.3), domain=(-1.0, 0.3, 1.0))
    actual = f.legcoeffs(5)
    expected = jnp.asarray([
        0.5450000000,
        -0.4365000000,
        0.5175625000,
        0.2173762500,
        -0.0574494375,
    ])
    assert actual.shape == (5,)
    assert _max_abs(actual - expected) < 10 * float(f.vscale) * EPS


def _piecewise_vector():
    return cj.chebfun(
        lambda x: jnp.stack([jnp.abs(x - 0.3) ** 3, jnp.exp(x)], axis=-1),
        domain=(-1.0, -0.5, 0.3, 0.5, 1.0))


EXPECTED_PIECEWISE_VECTOR = jnp.asarray([
    [0.38702500000000000000000, 1.175201193643801456882382],
    [-0.71513550000000000000000, 1.103638323514326964786571],
    [0.78877862500000000000000, 0.357814350647372460479053],
    [-0.24012341250000000000000, 0.070455633668489027815297],
    [0.09643353890625000000000, 0.009965128148869178524617],
])


def test_original_pass5_piecewise_array_valued():
    f = _piecewise_vector()
    actual = f.legcoeffs(5)
    assert actual.shape == (5, 2)
    assert _max_abs(actual - EXPECTED_PIECEWISE_VECTOR) \
        < 10 * float(f.vscale) * EPS


def test_original_pass6_piecewise_row_chebfun():
    f = _piecewise_vector()
    row = f.transpose()
    actual = row.legcoeffs(5)
    assert actual.shape == (5, 2)
    assert _max_abs(actual - EXPECTED_PIECEWISE_VECTOR) \
        < 10 * float(f.vscale) * EPS


def test_piecewise_without_n_requires_explicit_order():
    f = cj.chebfun(lambda x: jnp.abs(x - 0.3), domain=(-1.0, 0.3, 1.0))
    try:
        f.legcoeffs()
    except ValueError as exc:
        assert "piecewise" in str(exc).lower()
    else:
        raise AssertionError("piecewise legcoeffs without n must fail")


def test_array_valued_coefficients_pad_all_columns():
    f = cj.chebfun(
        lambda x: jnp.stack([1.0 + 0 * x, 2.0 + 0 * x], axis=-1))
    actual = f.legcoeffs(4)
    expected = jnp.asarray([[1.0, 2.0], [0.0, 0.0],
                            [0.0, 0.0], [0.0, 0.0]])
    assert actual.shape == (4, 2)
    assert _max_abs(actual - expected) == 0.0


def test_original_pass1_legendre_matrix():
    q = cj.legpoly(jnp.arange(5))
    difference = q.legcoeffs() - jnp.eye(5)
    assert float(jnp.linalg.norm(difference, ord='fro')) < 10 * float(q.vscale) * EPS


def test_original_pass2_legendre_linear_combination():
    q = cj.legpoly(jnp.arange(10))
    v = jnp.arange(1., 11.)[:, None]
    qv = q @ v
    assert _max_abs(qv.legcoeffs() - v) < 10 * float(qv.vscale) * EPS
