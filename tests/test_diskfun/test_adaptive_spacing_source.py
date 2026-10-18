"""Literal constructor getPoints/getTol controls, source pin 7574c77."""

import json
import math
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import (
    _disk_col_pts,
    _disk_row_pts,
    _get_tol,
    _phase_one_disk,
    diskfun,
)

RAW = json.loads(
    (Path(__file__).parents[1] / "fixtures/disk_adaptive_spacing_native_20261018.json").read_text()
)


def array(record):
    if math.prod(record["shape"]) == 0:
        return np.empty(record["shape"])
    return np.asarray([struct.unpack(">d", bytes.fromhex(v))[0] for v in record["re_hex"]]).reshape(
        record["shape"], order="F"
    )


def words(actual, record):
    expected = array(record)
    actual = np.asarray(actual, dtype=np.float64).reshape(expected.shape)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def operation(kind, theta, radius):
    if kind == "zero":
        return 0 * theta
    if kind == "constant":
        return 1 + 0 * theta
    if kind == "radial2":
        return radius * radius + 0 * theta
    if kind == "odd7":
        return radius * jnp.sin(7 * theta)
    order = int(kind[-1])
    return radius * radius * jnp.cos(order * theta)


@pytest.mark.parametrize("grid", RAW["grids"], ids=lambda c: str(c["grid"]))
def test_source_grid_coordinates(grid):
    g = grid["grid"]
    assert len(_disk_row_pts(g)) == 2 * g
    assert len(_disk_col_pts(g)) == g + 1
    words(_disk_row_pts(g), grid["theta"])
    words(_disk_col_pts(g), grid["radius"])


@pytest.mark.parametrize(
    "grid,case",
    [(g, c) for g in RAW["grids"] for c in g["cases"]],
    ids=[str(g["grid"]) + "_" + c["name"] for g in RAW["grids"] for c in g["cases"]],
)
def test_native_tolerance_on_actual_source_operands(grid, case):
    g = grid["grid"]
    # Cardinality 2*g does not change the literal native hx=2*pi/g call.
    tolerance, scale = _get_tol(array(case["values"]), 2 * np.pi / g, 1 / g, np.finfo(float).eps)
    words(tolerance, case["tol"])
    words(scale, case["vscale"])


@pytest.mark.parametrize("case", RAW["public"], ids=lambda c: c["name"])
def test_native_public_source_spacing(case):
    native = case["actual"]
    level = float(array(case["level"]).reshape(()))
    f = diskfun(lambda theta, radius: operation(case["kind"], theta, radius), "polar", "eps", level)
    assert len(f) == native["rank"]
    assert f.nonzero_poles == native["nonzero_poles"]
    words(jnp.stack([tech.coeffs for tech in f.cols], axis=1), native["cols"])
    words(f.pivots, native["pivots"])
    words(f.pivot_locations, native["locations"])
    assert tuple(f.idx_plus) == tuple(array(native["idx_plus"]).astype(int).reshape(-1) - 1)
    assert tuple(f.idx_minus) == tuple(array(native["idx_minus"]).astype(int).reshape(-1) - 1)
    assert [len(t.coeffs) for t in f.cols] == [native["cols"]["shape"][0]] * native["rank"]
    assert [len(t.coeffs) for t in f.rows] == [native["rows"]["shape"][0]] * native["rank"]
    # Original constructor polar/cartesian evaluation bound; unchanged.
    values = f(jnp.asarray([-math.pi, -0.7, 0.0, math.pi]), jnp.asarray([0.0, 0.25, 0.5, 1.0]))
    np.testing.assert_allclose(
        np.asarray(values).reshape(-1),
        array(native["values"]).reshape(-1),
        rtol=0,
        atol=2e3 * np.finfo(float).eps,
    )


@pytest.mark.parametrize("grid", RAW["grids"], ids=lambda c: str(c["grid"]))
def test_fixed_length_callable_uses_source_coordinates(grid):
    calls = []

    def callback(theta, radius):
        if theta.ndim == 2:
            calls.append((np.asarray(theta), np.asarray(radius)))
        return radius * radius + 0 * theta

    g = grid["grid"]
    result = diskfun(callback, "polar", [2 * g - 1, g])
    theta, radius = calls[-1]
    assert theta.shape == radius.shape == (g + 1, 2 * g)
    words(theta[0], grid["theta"])
    words(radius[:, 0], grid["radius"])
    np.testing.assert_allclose(
        np.asarray(result(jnp.asarray([0.0, 0.4]), jnp.asarray([0.2, 0.8]))),
        np.asarray([0.04, 0.64]),
        rtol=0,
        atol=2e3 * np.finfo(float).eps,
    )


@pytest.mark.parametrize(
    "grid,case",
    [(g, c) for g in RAW["grids"] for c in g["cases"]],
    ids=[str(g["grid"]) + "_" + c["name"] for g in RAW["grids"] for c in g["cases"]],
)
def test_native_phase_one_decisions_at_production_tolerance(grid, case):
    g = grid["grid"]
    tolerance = _get_tol(array(case["values"]), 2 * np.pi / g, 1 / g, np.finfo(float).eps)[0]
    indices, pivots, pole, happy = _phase_one_disk(array(case["values"]), tolerance, 100.0, 8.0)
    phase = case["phase"]
    np.testing.assert_array_equal(indices, array(phase["indices"]) - 1)
    words(pivots, phase["pivot_array"])
    assert pole == phase["remove_poles"]
    assert bool(happy) == bool(phase["happy"])


def test_independent_coupled_exact_update_still_exhausts_width():
    # C=0 and B has one nonzero away from the origin.
    # Fp=Fm=B/2, so the coupled update has rankCount=2 and zero residual.
    # Native width=min(3,4)/8=3/8: reaching width overrides that zero residual.
    values = np.zeros((3, 4))
    values[1, 2] = 2.0
    indices, pivots, pole, happy = _phase_one_disk(values, 0.0, 100.0, 8.0)
    np.testing.assert_array_equal(pivots, [[1.0, 1.0]])
    assert indices.shape == (1, 2)
    assert not pole and not happy


def test_independent_exact_zero_returns_before_width_override():
    indices, pivots, pole, happy = _phase_one_disk(np.zeros((3, 4)), 0.0, 100.0, 8.0)
    np.testing.assert_array_equal(pivots, [[0.0, 0.0]])
    assert indices.shape == (1, 2)
    assert not pole and happy
