"""All four original generalized Sylvester round-trip assertions.

Provenance
----------
MATLAB source : tests/chebop2/test_BartelsStewart.m
Chebfun commit: 7574c77
Inputs retain the original rng(0) draws and sequential real/complex cases.
The reference manifest records native capture provenance. No seed selection,
assertion omissions, or tolerance changes are used.
"""
from pathlib import Path

import numpy as np
import pytest

from chebfunjax.operators.chebop2 import bartels_stewart

_REFERENCE = Path(__file__).parents[2] / 'references/bartels_stewart_native.npz'


@pytest.mark.parametrize('case,multiplier', [(1, 1e4), (2, 1e5), (3, 1e8), (4, 1e8)])
def test_original_bartels_stewart_roundtrip(case, multiplier):
    with np.load(_REFERENCE, allow_pickle=False) as data:
        A, B, C, D, X = (data[f'case{case}_{key}'] for key in ['A', 'B', 'C', 'D', 'X'])
    E = A @ X @ B.T + C @ X @ D.T
    Y = bartels_stewart(A, B, C, D, E, False, False)
    # MATLAB norm(matrix) is the spectral norm, not the Frobenius norm.
    assert np.linalg.norm(Y-X, ord=2) < multiplier * np.finfo(np.float64).eps
