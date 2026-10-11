"""Supplemental native row/column Quasimatrix FLIPLR dispatch/storage.

Pin7574c77 @chebfun/fliplr.m. Independent literal Chebyshev coefficient
reflection T_k(-x)=(-1)^k T_k(x); no new numerical accuracy threshold.
"""

import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix


@pytest.mark.parametrize("row", [True, False])
def test_literal_collection_orientation_coefficients_and_pointvalues(row):
    f = cj.Chebfun.from_coeffs(np.asarray([1 + 2j, 3 - 4j])).set_point_values(
        np.asarray([9 + 10j, 11 - 12j])
    )
    g = cj.Chebfun.from_coeffs(np.asarray([5 - 6j, 7 + 8j, 2 - 1j])).set_point_values(
        np.asarray([13 - 14j, 15 + 16j])
    )
    q = Quasimatrix([f, g], f.domain)
    if row:
        q = q.T
    before = [np.asarray(c.point_values).copy() for c in q.cols]
    out = q.fliplr()
    assert out.is_transposed == row
    for actual, source in zip(out.cols, q.cols if row else q.cols[::-1]):
        coefficients = np.asarray(source.funs[0].tech.coeffs)
        expected = coefficients * ((-1.0) ** np.arange(len(coefficients))) if row else coefficients
        np.testing.assert_array_equal(actual.funs[0].tech.coeffs, expected)
        np.testing.assert_array_equal(
            actual.point_values,
            np.asarray(source.point_values)[::-1] if row else source.point_values,
        )
    for source, saved in zip(q.cols, before):
        np.testing.assert_array_equal(source.point_values, saved)
