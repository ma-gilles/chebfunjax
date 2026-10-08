"""Newton/recurrence Gauss--Laguerre REC and RECW rules from Chebfun.

The optional RECW flag retains the first exact-zero weight and stops.
The public Laguerre wrapper handles method selection,
normalization, interval mapping, and barycentric weights.

Provenance
----------
MATLAB source : lagpts.m (``lag_rec``)
Chebfun commit: 7574c77
Original author: Peter Opsomer, August 2017.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from chebfunjax.utils._gradual import gradual_positive_divide
from chebfunjax.utils.gamma_ratio import _gamma_ratio


def _recw_weight_is_zero(weight):
    """Source ``w == 0`` without CPU arithmetic flushing a subnormal."""
    bits = lax.bitcast_convert_type(weight, jnp.uint64)
    return (bits << jnp.uint64(1)) == jnp.uint64(0)


def _recw_raw_weight(ratio, z, derivative):
    """Separate source square, product and division, retaining overflow."""
    square = lax.optimization_barrier(derivative * derivative)
    denominator = lax.optimization_barrier(z * square)
    return gradual_positive_divide(ratio, denominator)


@partial(jax.jit, static_argnames=("n", "flag"))
def _lag_rec(n: int, alpha: float = 0.0, *, flag: bool = False):
    """Compute Gauss--Laguerre nodes and weights by source recurrence.

    Parameters
    ----------
    n : int
        Number of nodes. Zero returns empty arrays.
    alpha : float, default 0
        Generalized Laguerre parameter.
    flag : bool, default False
        RECW: stop at and include the first exact-zero weight.

    Returns
    -------
    x, w : tuple of jax.Array
        Ascending nodes and weights for ``x**alpha * exp(-x)``. The weights
        are the raw source recurrence weights. With ``flag=True``, return
        ``(x, w, length)`` in fixed n-capacity arrays; entries after length
        are zero padding. The public wrapper slices and normalizes.

    Provenance
    ----------
    MATLAB source : lagpts.m (``lag_rec``), gammaratio.m
    Chebfun commit: 7574c77
    """
    if n < 0:
        raise ValueError("lag_rec: n must be nonnegative")
    if n == 0:
        empty = jnp.empty((0,), dtype=jnp.float64)
        return (empty, empty, jnp.asarray(0, jnp.int32)) if flag else (empty, empty)

    alpha_value = jnp.asarray(alpha, dtype=jnp.float64)
    n_value = jnp.asarray(n, dtype=jnp.float64)
    initial_x = jnp.zeros((n,), dtype=jnp.float64)
    initial_w = jnp.zeros((n,), dtype=jnp.float64)

    def eval_poly(z):
        """Evaluate normalized L_n^alpha(z) and its derivative recurrence."""
        def recur(j, state):
            p1, p2 = state
            jf = jnp.asarray(j, dtype=jnp.float64)
            p1_next = (
                (-z + alpha_value + 2.0 * jf + 1.0) * p1
                - (alpha_value + jf) * p2
            ) / (jf + 1.0)
            return p1_next, p1

        p1, p2 = lax.fori_loop(
            0, n, recur,
            (jnp.asarray(1.0), jnp.asarray(0.0)),
        )
        derivative = (n_value * p1 - (n_value + alpha_value) * p2) / z
        return p1, derivative

    def root_step(i, arrays):
        roots, weights = arrays

        def guess_after_first(_):
            def second_guess(_):
                return (roots[0] + (15.0 + 6.25 * alpha_value)
                        / (1.0 + 0.9 * alpha_value + 2.5 * n_value))

            def later_guess(_):
                ai = jnp.asarray(i - 1, dtype=jnp.float64)
                step_scale = (
                    (1.0 + 2.55 * ai) / (1.9 * ai)
                    + 1.26 * ai * alpha_value / (1.0 + 3.5 * ai)
                ) / (1.0 + 0.3 * alpha_value)
                return (roots[i - 1]
                        + step_scale * (roots[i - 1] - roots[i - 2]))

            return lax.cond(i == 1, second_guess, later_guess, operand=None)

        def first_guess(_):
            return ((1.0 + alpha_value) * (3.0 + 0.92 * alpha_value)
                    / (1.0 + 2.4 * n_value + 1.8 * alpha_value))

        z0 = lax.cond(i == 0, first_guess, guess_after_first, operand=None)

        def newton_condition(state):
            count, _z, converged = state
            return (count < 10) & ~converged

        def newton_step(state):
            count, z, _converged = state
            p1, pp = eval_poly(z)
            z_next = z - p1 / pp
            converged = jnp.abs(z_next - z) < 1e-15
            return count + 1, z_next, converged

        _, z, _ = lax.while_loop(
            newton_condition,
            newton_step,
            (jnp.asarray(0, dtype=jnp.int32), z0,
             jnp.asarray(False)),
        )
        _, pp = eval_poly(z)
        weight = (_recw_raw_weight(_gamma_ratio(n + 1, alpha), z, pp) if flag
                  else _gamma_ratio(n + 1, alpha) / (z * pp**2))
        return roots.at[i].set(z), weights.at[i].set(weight)

    if flag:
        def continue_roots(state):
            i, _roots, _weights, stopped = state
            return (i < n) & ~stopped

        def next_root(state):
            i, roots, weights, _stopped = state
            roots, weights = root_step(i, (roots, weights))
            stopped = _recw_weight_is_zero(weights[i])
            # Store first, then stop: lag_rec includes the zero-weight node.
            return i + 1, roots, weights, stopped

        length, roots, weights, _ = lax.while_loop(
            continue_roots, next_root,
            (jnp.asarray(0, jnp.int32), initial_x, initial_w, jnp.asarray(False)))
        return roots, weights, length
    return lax.fori_loop(0, n, root_step, (initial_x, initial_w))
