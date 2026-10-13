"""Explicit-state JAX implementation of MATLAB's captured MT19937 uniform stream.

Provenance
----------
MATLAB source : ``rng`` documents the ``twister`` generator as MT19937;
                ``rand`` supplies the uniform stream.
Chebfun commit: ``7574c77680d7e82b79626300bf255498271a72df`` (native consumer pin)
Algorithm     : Matsumoto and Nishimura, MT19937, ACM TOMACS 8(1), 1998,
                DOI 10.1145/272991.272995. The implementation below is an
                independent translation of the published recurrence. Seed,
                output packing, and array ordering are qualified against the
                accompanying native MATLAB captures.

Only explicit-state uniform draws are covered. This module does not implement
MATLAB's global RNG, ``randn``, ``randi``, or ``randperm`` behavior.
"""
from __future__ import annotations

from math import prod
from typing import NamedTuple

import jax
import jax.numpy as jnp


class MatlabTwisterState(NamedTuple):
    """624 MT words and the next output index (624 requests a twist).

    Provenance
    ----------
    MATLAB source : ``rng`` state structure for ``twister`` (625 uint32 words).
    Chebfun commit: ``7574c77680d7e82b79626300bf255498271a72df``.
    """

    words: jax.Array
    index: jax.Array


def init_matlab_twister(seed: int) -> MatlabTwisterState:
    """Initialize an explicit MT19937 state from a MATLAB ``rng(seed,'twister')`` seed.

    Seed zero maps to 5489, as observed in the retained R2025b native capture;
    other supported uint32 seeds use the published MT19937 initialization
    recurrence. Seed must be a Python integer in ``[0, 2**32-1]``.

    Provenance
    ----------
    MATLAB source : ``rng``; seed/state behavior is captured in
                    ``matlab_uniform_twister_capture_20261014``.
    Chebfun commit: ``7574c77680d7e82b79626300bf255498271a72df``.
    Algorithm     : Matsumoto–Nishimura MT19937 initialization recurrence,
                    DOI 10.1145/272991.272995.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    if not jax.config.x64_enabled:
        # MATLAB uniform output is binary64; x64 is part of this API contract.
        raise ValueError("MATLAB Twister uniform output requires JAX x64")
    seed = 5489 if seed == 0 else seed
    u32 = jnp.uint32
    indices = jnp.arange(624, dtype=u32)

    def step(previous, i):
        current = u32(1812433253) * (previous ^ (previous >> u32(30))) + i
        return current, current

    first = u32(seed)
    _, tail = jax.lax.scan(step, first, indices[1:])
    words = jnp.concatenate((jnp.reshape(first, (1,)), tail))
    return MatlabTwisterState(words, jnp.asarray(624, dtype=jnp.int32))


def _twist(words: jax.Array) -> jax.Array:
    """Apply the published in-place MT19937 twist using JAX control flow."""
    u32 = jnp.uint32
    upper = u32(0x80000000)
    lower = u32(0x7FFFFFFF)
    matrix = u32(0x9908B0DF)

    def step(i, updated):
        next_i = jnp.where(i == 623, 0, i + 1)
        source_i = jnp.where(i < 227, i + 397, i - 227)
        combined = (updated[i] & upper) | (updated[next_i] & lower)
        source = updated[source_i]
        twisted = source ^ (combined >> u32(1)) ^ jnp.where(
            (combined & u32(1)) != 0, matrix, u32(0)
        )
        return updated.at[i].set(twisted)

    return jax.lax.fori_loop(0, 624, step, words)


def matlab_rand_uniform(
    state: MatlabTwisterState, shape: tuple[int, ...]
) -> tuple[jax.Array, MatlabTwisterState]:
    """Draw MATLAB-order uniform binary64 values and return the next state.

    The shape is a static Python tuple. Values are generated in stream order,
    consuming two MT uint32 outputs per binary64 via the captured 27+26-bit
    conversion. Multidimensional results use MATLAB column-major fill order.
    An empty dimension returns an empty array without advancing the state.
    ``shape=()`` requests one scalar value.

    Provenance
    ----------
    MATLAB source : built-in ``rand``; native vectors, states, and mixed-shape
                    outputs are retained under
                    ``matlab_uniform_twister_capture_20261014/native_v3``.
    Chebfun commit: ``7574c77680d7e82b79626300bf255498271a72df``.
    Algorithm     : MT19937 published recurrence plus captured 27+26-bit
                    uniform conversion; no NumPy or SciPy RNG is used.
    """
    if not jax.config.x64_enabled:
        raise ValueError("MATLAB Twister uniform output requires JAX x64")
    if not isinstance(shape, tuple) or any(not isinstance(d, int) or d < 0 for d in shape):
        raise ValueError("shape must be a tuple of nonnegative Python integers")
    count = prod(shape) if shape else 1
    if count == 0:
        return jnp.empty(shape, dtype=jnp.float64), state

    nwords = count * 2
    u32 = jnp.uint32

    def draw_one(carry, _):
        words, index = carry
        needs_twist = index == 624
        words = jax.lax.cond(needs_twist, _twist, lambda x: x, words)
        index = jnp.where(needs_twist, jnp.int32(0), index)
        value = words[index]
        value = value ^ (value >> u32(11))
        value = value ^ ((value << u32(7)) & u32(0x9D2C5680))
        value = value ^ ((value << u32(15)) & u32(0xEFC60000))
        value = value ^ (value >> u32(18))
        return (words, index + jnp.int32(1)), value

    (words, index), output_words = jax.lax.scan(
        draw_one, (state.words, state.index), xs=None, length=nwords
    )
    pairs = output_words.reshape((count, 2)).astype(jnp.uint64)
    high27 = pairs[:, 0] >> jnp.uint64(5)
    high26 = pairs[:, 1] >> jnp.uint64(6)
    packed = high27 * jnp.uint64(1 << 26) + high26
    linear = packed.astype(jnp.float64) * jnp.float64(1.0 / (1 << 53))
    if len(shape) > 1:
        reversed_shape = tuple(reversed(shape))
        axes = tuple(reversed(range(len(shape))))
        values = linear.reshape(reversed_shape).transpose(axes)
    else:
        values = linear.reshape(shape)
    return values, MatlabTwisterState(words, index)
