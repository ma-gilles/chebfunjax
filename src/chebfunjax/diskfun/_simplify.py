"""Fuse the existing disk slice simplification pipeline on CPU binary64.

Each slice retains its own source padding and FFT lengths. No common-length
padding is introduced before the noisy FFT round-trip. Only the completed
magnitudes are padded, for a shared cutoff kernel with the original active
length. The host adapter keeps
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

from chebfunjax.diskfun._chop_active import _standard_chop_active
from chebfunjax.tech.chebtech import Chebtech2, _prolong_coeffs
from chebfunjax.tech.trigtech import (
    Trigtech,
    _trig_prolong_coeffs,
    _trig_source_pairs_for_chop,
    trig_coeffs2vals,
    trig_vals2coeffs,
)
from chebfunjax.utils.transforms import coeffs2vals, vals2coeffs


@jax.jit
def _cheb_chop_input(coeffs):
    n = max(17, math.floor(coeffs.shape[0] * 1.25 + 5 + 0.5))
    noisy = vals2coeffs(coeffs2vals(_prolong_coeffs(coeffs, n)))
    magnitudes = jnp.asarray(jnp.abs(noisy), dtype=jnp.float64)
    capacity = 1 << (magnitudes.size - 1).bit_length()
    return jnp.pad(magnitudes, (0, capacity - magnitudes.size)), jnp.asarray(magnitudes.size)


@jax.jit
def _trig_chop_input(coeffs):
    n = max(17, math.floor(coeffs.shape[0] * 1.25 + 5 + 0.5))
    prolonged = _trig_prolong_coeffs(coeffs, n)
    noisy = trig_vals2coeffs(trig_coeffs2vals(jnp.abs(prolonged[::-1])))
    pairs = _trig_source_pairs_for_chop(noisy)
    magnitudes = jnp.asarray(jnp.abs(pairs), dtype=jnp.float64)
    capacity = 1 << (magnitudes.size - 1).bit_length()
    return jnp.pad(magnitudes, (0, capacity - magnitudes.size)), jnp.asarray(magnitudes.size)


def _cheb_cutoff(coeffs, tol):
    magnitudes, length = _cheb_chop_input(coeffs)
    return _standard_chop_active(magnitudes, tol, length)


def _trig_cutoff(coeffs, tol):
    magnitudes, length = _trig_chop_input(coeffs)
    return _standard_chop_active(magnitudes, tol, length)


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


@jax.jit
def _slice_scale_kernel(tech):
    return jnp.max(jnp.abs(jnp.asarray(tech.values)))


def slice_scale(tech):
    """Fuse the existing scale expression for qualified scalar CPU slices.

    Provenance: @chebfun/simplify.m obtains local vscales before setting
    global tolerances. The existing tech.values property, including a stored
    Trigtech value array, remains the source of values here.
    """
    is_cheb = type(tech) is Chebtech2 and tech.coeffs.dtype == jnp.float64
    is_trig = (
        type(tech) is Trigtech
        and tech.is_real
        and tech.coeffs.dtype in (jnp.float64, jnp.complex128)
    )
    if (
        (is_cheb or is_trig)
        and jax.default_backend() == "cpu"
        and tech.coeffs.ndim == 1
        and tech.n > 0
    ):
        return float(_slice_scale_kernel(tech))
    return float(jnp.max(jnp.abs(jnp.asarray(tech.values))))
