"""Fuse the existing disk slice simplification pipeline on CPU binary64.

Each slice retains its own source padding and FFT lengths. No common-length
padding is introduced before the noisy FFT round-trip. The host adapter keeps
source output-length and coefficient-storage decisions unchanged.

Provenance
----------
Chebfun pin7574c77680d7e82b79626300bf255498271a72df:
@chebtech/simplify.m, @trigtech/simplify.m, and standardChop.m.
This module fuses the already translated operations for disk's scalar slices;
other technologies, shapes, dtypes and backends retain their original method.
"""

import math

import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech2, _prolong_coeffs
from chebfunjax.tech.trigtech import (
    Trigtech,
    _trig_prolong_coeffs,
    _trig_source_pairs_for_chop,
    trig_coeffs2vals,
    trig_vals2coeffs,
)
from chebfunjax.utils.misc import _standard_chop_kernel
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs


@jax.jit
def _cheb_cutoff(coeffs, tol):
    n = max(17, math.floor(coeffs.shape[0] * 1.25 + 5 + 0.5))
    noisy = vals2coeffs(coeffs2vals(_prolong_coeffs(coeffs, n)))
    return _standard_chop_kernel(noisy, tol)


@jax.jit
def _trig_cutoff(coeffs, tol):
    n = max(17, math.floor(coeffs.shape[0] * 1.25 + 5 + 0.5))
    prolonged = _trig_prolong_coeffs(coeffs, n)
    noisy = trig_vals2coeffs(trig_coeffs2vals(jnp.abs(prolonged[::-1])))
    return _standard_chop_kernel(_trig_source_pairs_for_chop(noisy), tol)


def simplify_slice(tech, tol):
    """Preserve the original method outside qualified scalar CPU slice types."""
    is_cheb = type(tech) is Chebtech2 and tech.coeffs.dtype == jnp.float64
    is_trig = (
        type(tech) is Trigtech
        and tech.is_real
        and tech.coeffs.dtype in (jnp.float64, jnp.complex128)
    )
    if (
        not (is_cheb or is_trig)
        or jax.default_backend() != "cpu"
        or tech.coeffs.ndim != 1
        or not tech.ishappy
        or tech.n == 0
        or not 0 < tol < 1
    ):
        return tech.simplify(tol)

    if is_cheb:
        cutoff = min(int(_cheb_cutoff(tech.coeffs, jnp.asarray(tol))), tech.n)
        return Chebtech2(coeffs=tech.coeffs[:cutoff], ishappy=tech.ishappy)

    cutoff = min(int(_trig_cutoff(tech.coeffs, jnp.asarray(tol))), tech.n)
    n_keep = max(1, cutoff + 1 if cutoff % 2 == 0 else cutoff)
    coeffs = _trig_prolong_coeffs(tech.coeffs, n_keep)
    if jnp.iscomplexobj(coeffs) and bool(jnp.all(jnp.imag(coeffs) == 0)):
        coeffs = jnp.real(coeffs)
    return Trigtech(
        coeffs=coeffs,
        is_real=tech.is_real,
        real_columns=tech.real_columns,
        ishappy=tech.ishappy,
    )
