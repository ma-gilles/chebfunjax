"""Exact column assembly controls, independent NumPy stacking oracle.

Provenance: source coefficient matrix assembly; grouping changes copies only,
not the existing Fourier prolongation. No page/performance qualification.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun import _bmci, _plus
from chebfunjax.tech.trigtech import _trig_prolong_coeffs


def factors(rank, complex_values, unequal):
    arrays = []
    for column in range(rank):
        length = (5+column % 5) if unequal else 9
        values = ((np.arange(length)-3)*(column+1)).astype(np.float64)
        values[0], values[-1] = 0.0, -0.0
        if complex_values:
            out = np.empty(length, dtype=np.complex128)
            out.real = values
            out.imag = -values[::-1]
            values = out
        arrays.append(jnp.asarray(values))
    return [SimpleNamespace(coeffs=x) for x in arrays]


def assert_exact(actual, expected):
    actual = np.asarray(actual)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('rank', [1, 31, 32, 33, 185, 714])
@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('unequal', [False, True])
def test_both_source_factor_assemblies(rank, complex_values, unequal):
    techs = factors(rank, complex_values, unequal)
    size = max(t.coeffs.shape[0] for t in techs)
    # Isolate changed assembly: same qualified source per-factor prolongation,
    # independently stacked on host. Signed zero byte patterns are included.
    expected = np.stack([np.asarray(_trig_prolong_coeffs(t.coeffs, size))
                         for t in techs], axis=1)
    for assemble in (_bmci._stack, _plus._stack):
        assert_exact(assemble(techs), expected)


def test_factor_assembly_jit_preserves_complex_columns():
    techs = factors(33, True, True)
    arrays = tuple(t.coeffs for t in techs)
    expected = np.stack([np.asarray(_trig_prolong_coeffs(t.coeffs, 9))
                         for t in techs], axis=1)

    def compiled(values):
        return _bmci._stack([SimpleNamespace(coeffs=x) for x in values])

    assert_exact(jax.jit(compiled)(arrays), expected)
