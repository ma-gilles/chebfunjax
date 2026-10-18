"""Native C1/C2 transform stages and ABS for finite and infinite inputs.

Reference: Chebfun7574c77 @chebtech1/vals2coeffs.m and coeffs2vals.m,
@chebtech2/vals2coeffs.m and coeffs2vals.m, @chebtech/abs.m.
Finite comparison retains the original C1 transform tests'100eps bound.
Nonfinite components and infinity signs are compared separately and exactly.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_FIXTURES = Path(__file__).with_name('fixtures')
_META = json.loads((_FIXTURES/'nonfinite_transform_stages_native.json').read_text())


@pytest.mark.parametrize('index', range(8), ids=[r['id'] for r in _META['rows']])
@pytest.mark.parametrize('stage', ['coeffs2vals', 'vals2coeffs_native_magnitudes', 'abs'])
def test_native_transform_stage_components(index, stage):
    row = _META['rows'][index]
    tech = Chebtech1 if row['tech'] == 1 else Chebtech2
    with np.load(_FIXTURES/'nonfinite_transform_stages_native.npz', allow_pickle=False) as data:
        coeffs = jnp.asarray(data[f'{index}_input'])
        if stage == 'coeffs2vals':
            actual = tech.coeffs2vals(coeffs)
            expected = data[f'{index}_values']
        elif stage == 'vals2coeffs_native_magnitudes':
            actual = tech.vals2coeffs(jnp.asarray(data[f'{index}_magnitudes']))
            expected = data[f'{index}_coefficients']
        else:
            actual = abs(tech(coeffs=coeffs, ishappy=False)).coeffs
            expected = data[f'{index}_coefficients']
        actual = np.asarray(actual)
        assert actual.shape == expected.shape
        for component in (np.real, np.imag):
            a, b = component(actual), component(expected)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_array_equal(np.isposinf(a), np.isposinf(b))
            np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
            finite = np.isfinite(b)
            np.testing.assert_allclose(a[finite], b[finite], rtol=0,
                                       atol=100*np.finfo(np.float64).eps)
