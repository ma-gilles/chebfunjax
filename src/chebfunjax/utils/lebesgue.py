"""Sampled adapters and the continuous source Lebesgue constructor.

The sampled helpers retain their equispaced grid and approximate constant API.
Their array results are JAX arrays; the continuous constructor is ``lebesgue``.

Provenance
----------
MATLAB source: lebesgue.m, baryWeights.m.
Chebfun commit: 7574c77
Original: Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.utils.lebesgue_source import _values, lebesgue

__all__ = ["lebesgue", "lebesgue_function", "lebesgue_constant", "bary_weights"]


def bary_weights(x):
    """Return unnormalised polynomial weights for the sampled adapter.

    This preserves the legacy helper's reciprocal product convention. The
    continuous constructor uses the source capacity scaled, normalised weights.

    Provenance
    ----------
    MATLAB source: baryWeights.m (unnormalised Python compatibility adapter).
    Chebfun commit: 7574c77
    """
    x = jnp.asarray(x, dtype=jnp.float64).ravel()
    differences = x[:, None] - x[None, :]
    differences = differences.at[jnp.diag_indices(x.size)].set(1.0)
    return jnp.prod(jnp.sign(differences), axis=1) * jnp.exp(
        -jnp.sum(jnp.log(jnp.abs(differences)), axis=1)
    )


def _lebesgue_fun_at(t, x, w):
    """Evaluate the sampled adapter at one point using the JAX source formula."""
    return _values(jnp.asarray(t), jnp.asarray(x), jnp.asarray(w))


def lebesgue_function(nodes, domain=(-1.0, 1.0), *, n_eval=2001):
    """Return an equispaced grid and its sampled polynomial Lebesgue values.

    Provenance
    ----------
    MATLAB source: lebesgue.m (polyLebesgueFun; sampled Python adapter).
    Chebfun commit: 7574c77
    """
    nodes = jnp.asarray(nodes, dtype=jnp.float64).ravel()
    t = jnp.linspace(float(domain[0]), float(domain[1]), n_eval)
    return t, _values(t, nodes, bary_weights(nodes))


def lebesgue_constant(nodes, domain=(-1.0, 1.0), *, n_eval=2001):
    """Return the approximate maximum over the sampled adapter's grid.

    Use ``lebesgue(..., return_constant=True)`` for the continuous inf norm.

    Provenance
    ----------
    MATLAB source: lebesgue.m (norm(L, inf); sampled Python adapter).
    Chebfun commit: 7574c77
    """
    _, values = lebesgue_function(nodes, domain, n_eval=n_eval)
    return float(jnp.max(values))
