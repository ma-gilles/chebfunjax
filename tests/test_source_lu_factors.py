"""Native factor corpus for source real binary64 LU.

Provenance
----------
MATLAB source: @valsDiscretization/mldivide.m, native dense LU calls
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Capture hashes and scope are in data/source_dense_lu/manifest.json.
"""
import gc
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils._source_lu_solve import _factors

_DATA = Path(__file__).parent/'data/source_dense_lu'
_MANIFEST = json.loads((_DATA/'manifest.json').read_text())
_DIMENSIONS = sorted({case['n'] for case in _MANIFEST['cases']})


@pytest.mark.parametrize('n', _DIMENSIONS)
def test_native_factor_corpus(n):
    # Bound compilation-cache retention between dimension groups on CI.
    # Every captured case executes; no assertion or input domain is masked.
    jax.clear_caches()
    gc.collect()
    archive = _DATA/'native_controls.npz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == _MANIFEST['npz_sha256']
    try:
        with np.load(archive, allow_pickle=False) as data:
            for case in _MANIFEST['cases']:
                if case['n'] != n:
                    continue
                key = case['key']
                packed, permutation = _factors(jnp.asarray(data[key+'_matrix']))
                actual = np.asarray(packed)
                expected = data[key+'_packed']
                np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected), err_msg=case['name'])
                non_nan = ~np.isnan(expected)
                np.testing.assert_array_equal(actual.view(np.uint64)[non_nan],
                                              expected.view(np.uint64)[non_nan], err_msg=case['name'])
                np.testing.assert_array_equal(np.asarray(permutation),
                                              data[key+'_permutation'], err_msg=case['name'])
    finally:
        jax.clear_caches()
        gc.collect()
