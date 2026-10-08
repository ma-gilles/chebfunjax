"""Explicit small Hermite source outcomes, including source numerical defects.

Provenance
----------
MATLAB source: hermpts.m, HermiteInitialGuesses/hermpts_rec/hermpts_asy.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Fresh MATLAB fixture records raw real and imaginary binary64 components.
"""
# uses-numpy: decode captured MATLAB outputs and assert source predicates.
import json
from pathlib import Path

import numpy as np
import pytest

from chebfunjax.utils.quadrature import hermpts

REFERENCE = json.loads((Path(__file__).parent / 'fixtures' /
                        'hermite_small_matlab.json').read_text())
TOL = 10 * np.finfo(np.float64).eps


def _decode(value):
    def component(key):
        values = value[key]
        if isinstance(values, str):
            values = [values]
        return np.asarray([np.frombuffer(bytes.fromhex(v), dtype='>f8')[0]
                           for v in values if v])
    real, imag = component('real_hex'), component('imag_hex')
    return real, imag


@pytest.mark.parametrize('method,n,kind', [(method, n, kind)
    for method in ['REC'] for n in range(2, 21)
    for kind in ['phys', 'prob']])
def test_explicit_small_source(method, n, kind):
    ref = next(r for r in REFERENCE['cases'] if (r['method'], r['n'], r['kind'])
               == (method, n, kind))
    if ref['error_message']:
        assert n == 3 and 'concatenat' in ref['error_message']
        with pytest.raises(ValueError):
            hermpts(n, kind, method=method, bary=True)
        return
    actual = hermpts(n, kind, method=method, bary=True)
    # Preserve original test_hermpts bounds: REC42 node10tol, weight/v tol;
    # ASY251 node4tol, weight10tol, v100tol. No small-order moment claim.
    bounds = [10*TOL, TOL, TOL] if method == 'REC' else [4*TOL, 10*TOL, 100*TOL]
    for name, got, bound in zip(['x', 'w', 'v'], actual, bounds):
        real, imag = _decode(ref[name])
        got = np.asarray(got)
        assert got.shape == real.shape == imag.shape == (n,)
        for part, expected in [(got.real, real), (got.imag, imag)]:
            np.testing.assert_array_equal(np.isnan(part), np.isnan(expected))
            np.testing.assert_array_equal(np.isposinf(part), np.isposinf(expected))
            np.testing.assert_array_equal(np.isneginf(part), np.isneginf(expected))
            finite = np.isfinite(expected)
            np.testing.assert_allclose(part[finite], expected[finite], rtol=0, atol=bound)
