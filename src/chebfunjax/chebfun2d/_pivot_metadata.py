"""Retained native pivotValues alongside the legacy reciprocal CDR weights.

None means no original/source-defined pivot values are available. Rounded
reciprocal recovery is never labeled retained metadata.

Valid objects keep weights equal to native CDR(raw) whenever raw is present.
Raw pivots are authoritative for source arithmetic. Direct replacement of only
one dependent leaf violates this invariant; getters reject such stale objects.
Private scalar helpers require a static Python scalar, not a traced scalar.

Provenance
----------
MATLAB source : @separableApprox/{cdr,uminus,mtimes,pivots}.m
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp

from chebfunjax.chebfun2d._numeric_constructor import _source_inf_magnitude


def _cdr_weights(pivot_values):
    """Native CDR reciprocal followed by infinite-reciprocal replacement."""
    weights = 1 / jnp.asarray(pivot_values)
    return jnp.where(_source_inf_magnitude(weights), 0, weights)


def _retained_pivots(approx):
    """Eager source accessor, rejecting stale metadata rather than guessing.

    Ordinary JIT/AD evaluation continues to use the existing weights. This
    eager check is used by source extrema and the explicit public raw getter.
    """
    raw = getattr(approx, "pivot_values", None)
    if raw is None:
        return None
    if raw.shape != approx.pivots.shape or raw.ndim != 1:
        raise ValueError("Native pivot metadata has incompatible shape")
    if not bool(jnp.array_equal(_cdr_weights(raw), approx.pivots, equal_nan=True)):
        raise ValueError("Native pivot metadata disagrees with CDR weights")
    return raw


def _negate(approx):
    """Native unary minus negates pivotValues, preserving factor metadata."""
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox

    raw = None if approx.pivot_values is None else -approx.pivot_values
    weights = -approx.pivots if raw is None else _cdr_weights(raw)
    return SeparableApprox(cols=list(approx.cols), rows=list(approx.rows),
        pivots=weights, pivot_values=raw, domain=approx.domain,
        techs=approx.techs, pivot_locations=approx.pivot_locations)


def _scale(approx, scalar):
    """Native p/scalar and exact zero branch; unknown weights stay unknown."""
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox

    if scalar == 0:
        raw = jnp.asarray([jnp.inf])
        return SeparableApprox(cols=[0*approx.cols[0]], rows=[0*approx.rows[0]],
            pivots=_cdr_weights(raw), pivot_values=raw, domain=approx.domain,
            techs=approx.techs, pivot_locations=approx.pivot_locations[:1])
    raw = None
    if approx.pivot_values is not None:
        # Native p/scalar must not become p*(1/scalar) for a static scalar.
        # Match the accepted colleague helper's full-shaped divisor boundary.
        divisor = jax.lax.optimization_barrier(
            jnp.broadcast_to(jnp.asarray(scalar), approx.pivot_values.shape))
        raw = jax.lax.optimization_barrier(approx.pivot_values/divisor)
    weights = approx.pivots*scalar if raw is None else _cdr_weights(raw)
    return SeparableApprox(cols=list(approx.cols), rows=list(approx.rows),
        pivots=weights, pivot_values=raw, domain=approx.domain,
        techs=approx.techs, pivot_locations=approx.pivot_locations)
