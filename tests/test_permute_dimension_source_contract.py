"""Supplemental native-captured validation/storage controls, not native slots."""

import json
from pathlib import Path

import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

FIX = Path(__file__).parent / "fixtures" / "permute_native_observations"
CASES = json.loads(FIX.with_suffix(".json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c["case"]))
def test_captured_numeric_orders(case):
    with np.load(FIX.with_suffix(".npz")) as data:
        order = data[f"order_{case['case']}"]
    f = cj.chebfun(lambda x: 1 + 2j + (3 - 4j) * x)
    before = np.asarray(f.chebcoeffs()).copy()
    points = np.asarray(f.point_values).copy()
    if not case["success"]:
        with pytest.raises(ValueError, match=case["identifier"]):
            f.permute(order)
    else:
        g = f.permute(order)
        assert g.is_transposed == case["is_transposed"]
        np.testing.assert_array_equal(g.chebcoeffs(), before)
        np.testing.assert_array_equal(g.point_values, points)
    np.testing.assert_array_equal(f.chebcoeffs(), before)
    np.testing.assert_array_equal(f.point_values, points)
    assert not f.is_transposed


def test_collection_nonconjugate_storage():
    f = cj.chebfun(lambda x: 1 + 2j + (3 - 4j) * x)
    q = Quasimatrix([f, 2 * f], f.domain)
    assert q.permute([1, 2, 3]) is q
    row = q.permute([2, 1, 3])
    assert row.is_transposed
    assert not row.permute([2, 1]).is_transposed
    for old, new in zip(q.cols, row.cols):
        np.testing.assert_array_equal(old.chebcoeffs(), new.chebcoeffs())
        np.testing.assert_array_equal(old.point_values, new.point_values)


def test_explicit_point_value_storage():
    f = cj.chebfun(lambda x: 1 + 2j + (3 - 4j) * x)
    f = f.set_point_values([[7 + 8j], [9 - 10j]])
    before = np.asarray(f.point_values).copy()
    row = f.permute([2, 1, 3])
    np.testing.assert_array_equal(row.point_values, before)
    np.testing.assert_array_equal(f.point_values, before)
    assert row._point_values is f._point_values
