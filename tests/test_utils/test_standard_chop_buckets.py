"""Bucket storage must preserve standardChop's source active length.

Provenance: standardChop.m, Chebfun commit 7574c77. These supplemental
regressions compare an unpadded source kernel and exercise bucket boundaries;
existing native-reference tests remain unchanged.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.misc import _standard_chop_kernel, standard_chop


@pytest.mark.parametrize("n", [17,31,32,33,63,64,65,127,128,129])
def test_active_length_cutoffs(n):
    rng = np.random.default_rng(n)
    vectors = [np.zeros(n), np.ones(n), .1**np.arange(n),
               np.exp(-np.arange(n)/3)+rng.normal(size=n)*1e-15,
               np.r_[np.ones(n-1), np.nan],
               np.r_[np.inf, np.ones(n-1)]]
    for coeffs in vectors:
        for tol in [np.finfo(float).eps, 1e-300]:
            expected = int(_standard_chop_kernel(jnp.asarray(coeffs), jnp.asarray(tol)))
            assert standard_chop(coeffs, tol) == expected

@pytest.mark.parametrize("dtype", [jnp.float32,jnp.float64,jnp.complex64,jnp.complex128])
def test_dtype_and_complex_envelope(dtype):
    coeffs = jnp.asarray(.1**np.arange(33), dtype=dtype)
    if jnp.issubdtype(dtype,jnp.complexfloating):
        coeffs = coeffs*(1+2j)
    expected = int(_standard_chop_kernel(coeffs,jnp.asarray(2.220446049250313e-16)))
    assert standard_chop(coeffs) == expected

def test_padding_does_not_invent_plateau():
    coeffs = jnp.asarray(np.r_[np.ones(10), np.full(7,1e-18)])
    assert standard_chop(coeffs) == 17
    assert int(_standard_chop_kernel(jnp.pad(coeffs,(0,15)),jnp.asarray(2.220446049250313e-16))) < 17

@pytest.mark.parametrize("n", [0,1,16])
def test_short_source_return_precedes_zero_envelope(n):
    assert standard_chop(jnp.zeros(n)) == n
    assert standard_chop(jnp.zeros(n),1.) == 1

@pytest.mark.parametrize("dtype", [jnp.int32,jnp.int64])
def test_padding_preserves_existing_integer_abs_overflow(dtype):
    # This is compatibility with the existing kernel, not a new integer policy.
    coeffs = jnp.full(17,jnp.iinfo(dtype).min,dtype=dtype)
    expected = int(_standard_chop_kernel(coeffs,jnp.asarray(2.220446049250313e-16)))
    assert expected == 17
    assert standard_chop(coeffs) == expected
