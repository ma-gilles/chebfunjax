"""Fresh native polynomial zero target status and IEEE normalization.

Source minimax.m uses norm(f)==0 without an EPS floor. Its0/0 normalized
correction is NaN; the perturbed exchange error is literal1e-19.
"""

import json
import struct
import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax import chebfun
from chebfunjax.utils.minimax import minimax


@pytest.mark.parametrize('input_kind', ['chebfun', 'callable'])
def test_native_zero_error_and_normalized_correction(input_kind):
    native = json.loads((Path(__file__).parent / 'fixtures' /
                         'minimax_zero_native_20261018.json').read_text())['zero']
    f = chebfun(0) if input_kind == 'chebfun' else lambda x: jnp.zeros_like(x)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        result = minimax(f, 1, max_iter=1)
    assert not emitted
    assert jnp.array_equal(result.coeffs, jnp.zeros_like(result.coeffs))
    assert result.iter == 1
    assert struct.pack('>d', result.err).hex() == native['error']['real_words'][0]
    assert struct.pack('>d', result.delta).hex() == native['delta']['real_words'][0]
