"""Singular-parameter outcomes captured from MATLAB lagpts.m, Chebfun7574c77.

Python keeps its documented one-dimensional output convention, including EXPW
where source implicit expansion yields a square barycentric matrix. Source
error identifiers are retained in the fixture; Python exception types differ.
"""
# uses-numpy: decode captured MATLAB binary64 outputs and compare host assertions.
import json
from pathlib import Path

import numpy as np
import pytest

from chebfunjax.utils.quadrature import lagpts

FIXTURE = json.loads((Path(__file__).parent / 'fixtures' /
                      'laguerre_alpha_minus_one_matlab.json').read_text())


def _values(encoded):
    return np.asarray([np.frombuffer(bytes.fromhex(v), dtype='>f8')[0]
                       for v in encoded['hex']])


@pytest.mark.parametrize('method,n', [(method, n) for method in
    ['default', 'REC', 'RECW', 'GW', 'GLR', 'RH', 'RHW', 'EXP', 'EXPW']
    for n in [0, 1, 2, 8, 42]])
def test_singular_source_outcomes(method, n):
    reference = next(r for r in FIXTURE if r['method'] == method and r['n'] == n)
    if reference['error_message']:
        # RHW's literal dispatcher rejects its name; RH fails at a zero-seed
        # negative-index lookup. Python currently rejects the singular alpha.
        with pytest.raises(ValueError):
            lagpts(n, alpha=-1, method=method, bary=True)
        return
    actual = lagpts(n, alpha=-1, method=method, bary=True)
    for key, value in zip(['x', 'w', 'v'], actual):
        expected = _values(reference[key])
        if key == 'v' and method == 'EXPW' and n:
            # Existing public 1D adapter: first source column; all columns are
            # NaN here. Do not claim literal source output-shape parity.
            expected = expected[:n]
        value = np.asarray(value)
        assert value.shape == expected.shape
        np.testing.assert_array_equal(np.isnan(value), np.isnan(expected))
        np.testing.assert_array_equal(np.isposinf(value), np.isposinf(expected))
        np.testing.assert_array_equal(np.isneginf(value), np.isneginf(expected))
        finite = np.isfinite(expected)
        # Retain the previously qualified EXP source node bound, without
        # demanding finite moments from this singular measure.
        np.testing.assert_allclose(value[finite], expected[finite], rtol=0, atol=8e-10)
