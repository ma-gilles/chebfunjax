"""Pinned Chebfun3 seed/optimizer/Newton transaction.

Provenance
----------
MATLAB source: @chebfun3/minandmax3.m, Chebfun commit7574c77.
Copyright2017 University of Oxford and the Chebfun Developers.
"""
import jax
import jax.numpy as jnp

from chebfunjax.utils._active_set_box_sqp import active_set_box
from chebfunjax.utils._active_set_finite_difference import finite_difference
from chebfunjax.utils._fminsearch import fminsearch
from chebfunjax.utils.quadrature import chebpts


def _forward(t, a, b):
    return b*(t+1)/2+a*(1-t)/2


def _inverse(x, a, b):
    return (x-a)/(b-a)-(b-x)/(b-a)


def source_seed(f):
    """Length-derived C2 grid, mode-order contractions, F-order first ties."""
    lengths = f.length()
    if f.ndf() > 5e4:
        lengths = tuple(min(n, 129) for n in lengths)
    nodes, factors = [], []
    for axis, (n, group) in enumerate(zip(lengths, (f.cols, f.rows, f.tubes))):
        a, b = f.domain[2*axis:2*axis+2]
        t = chebpts(n)
        x = t if (a, b) == (-1, 1) else _forward(t, a, b)
        nodes.append(x)
        factors.append(jnp.stack([factor(_inverse(x, a, b)) for factor in group], axis=1))
    values = f.core
    for axis, matrix in enumerate(factors):
        values = jnp.moveaxis(jnp.tensordot(matrix, values, axes=(1, axis)), 0, axis)
    flat = values.reshape(-1, order='F')
    if jnp.iscomplexobj(flat):
        raise TypeError('Complex Chebfun3 extrema ordering is not qualified')
    indices = jnp.stack((jnp.argmin(flat), jnp.argmax(flat)))
    m, n, _ = lengths
    locations = jnp.stack((nodes[0][indices % m], nodes[1][(indices//m) % n],
                           nodes[2][indices//(m*n)]), axis=1)
    return locations


def _outside(locations, lower, upper):
    return bool(jnp.any((locations <= lower) | (locations >= upper)))


def extrema(f, *, optimizer='active-set'):
    """Native two seeds, explicit optimizer branch, one Newton transaction.

    Provenance
    ----------
    MATLAB source: @chebfun3/minandmax3.m; Chebfun commit7574c77.
    Arithmetic follows readable R2017a sources; direct native controls use
    R2025b. The attempted legacy overlay failed before objective evaluation.

    Python adapters: extrema values remain a length-two vector. The native
    successful Newton assignments expand its initial column into a 2-by-2
    array. Optimizer errors propagate; the native fallback catch can instead
    leave locations undefined. Neither accidental behavior is reproduced.
    """
    if f.isempty():
        return jnp.empty((0,)), jnp.empty((0,))
    locations = source_seed(f)
    lower = jnp.asarray(f.domain[::2], dtype=jnp.float64)
    upper = jnp.asarray(f.domain[1::2], dtype=jnp.float64)
    objective = jax.jit(lambda p: f(p[0], p[1], p[2]))
    result_locations, result_values = [], []
    for i, sign in enumerate((1., -1.)):
        def target(p):
            return sign*objective(p)
        if optimizer == 'active-set':
            result = active_set_box(target, locations[i], lower, upper,
                                     finite_difference=finite_difference)
            point, value = result.x, sign*result.value
        elif optimizer == 'fminsearch':
            initial = jnp.arcsin(_inverse(locations[i], lower, upper))
            def mapped(p):
                return target(_forward(jnp.sin(p), lower, upper))
            result = fminsearch(mapped, initial, max_iterations=600,
                                 max_evaluations=600)
            point, value = _forward(jnp.sin(result.x), lower, upper), sign*result.fun
        else:
            raise ValueError('Unknown extrema optimizer')
        result_locations.append(point)
        result_values.append(value)
    locations, values = jnp.stack(result_locations), jnp.stack(result_values)
    if _outside(locations, lower, upper):
        return values, locations
    gradients = [f.diff(dim=axis+1) for axis in range(3)]
    hessians = [[f.diff(dim=row+1,k=2) if row == col else
                 gradients[col].diff(dim=row+1) for col in range(3)] for row in range(3)]
    proposals = []
    for point in locations:
        gradient = jnp.stack([g(*point) for g in gradients])
        hessian = jnp.stack([jnp.stack([h(*point) for h in row]) for row in hessians])
        delta = jnp.linalg.solve(hessian, -gradient)
        # Exact singular backend result: no fitted condition/rank threshold.
        if not bool(jnp.all(jnp.isfinite(delta))):
            return values, locations
        proposals.append(point+delta)
    proposed = jnp.stack(proposals)
    if _outside(proposed, lower, upper):
        return values, locations
    return jnp.stack([objective(p) for p in proposed]), proposed
