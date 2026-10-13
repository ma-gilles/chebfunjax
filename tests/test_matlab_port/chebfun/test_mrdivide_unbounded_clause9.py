"""Literal unbounded Chebfun array least-squares predicate, native clause 9.

Provenance
----------
MATLAB source : tests/chebfun/test_mrdivide.m, clauses 1–9
Chebfun commit: 7574c77
Native input/output capture: mrdivide9_source_capture_20261014/capture.mat

The source supplies the actual second seedRNG(6178) draw after constructing
clauses 1–8. Python's two-argument ``mrdivide`` is the least-squares route
corresponding to MATLAB ``mrdivide(A, B, 'ls')``.
"""
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import mrdivide


def test_native_mrdivide_clause9_unbounded_array_scalar_ls():
    fixture = np.load(Path(__file__).parent / 'fixtures' / 'mrdivide9_native_values.npz')
    x = jnp.asarray(fixture['x'])

    # Original source constructor and domain: one unbounded three-component
    # Chebfun, with a scalar right denominator through the LS entry point.
    domain = (-math.inf, 3.0 * math.pi)
    op = lambda t: jnp.stack(
        (jnp.exp(t), t * jnp.exp(t), (1.0 - jnp.exp(t)) / t), axis=-1
    )
    A = cj.chebfun(op, domain=domain)
    X = mrdivide(A, 3.0)

    # Keep the original MATLAB expression order and matrix infinity norm:
    # for this 100x3 error matrix, norm(err, inf) is max row sum(abs(err)).
    exact = jnp.concatenate(
        (jnp.exp(x) / 3.0,
         x * jnp.exp(x) / 3.0,
         (1.0 - jnp.exp(x)) / (3.0 * x)),
        axis=1,
    )
    values = X(x)
    err = values - exact
    source_bound = 1e2 * float(jnp.max(X.vscale)) * float(jnp.finfo(jnp.float64).eps)
    assert values.shape == (100, 3)
    assert np.linalg.norm(np.asarray(err), ord=np.inf) < source_bound

