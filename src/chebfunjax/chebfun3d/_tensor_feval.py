"""Native three-dimensional tensor-grid evaluation for Chebfun3.

Provenance
----------
MATLAB source: @chebfun3/feval.m, tensorCase, Chebfun commit7574c77.
Original authors: Copyright2017 The University of Oxford and Chebfun Developers.
Eager grid recognition follows source order and exact zero comparisons.
Traced, matrix, vector and unstructured inputs retain the existing evaluator.
"""
import jax
import jax.numpy as jnp

from ._mtimes import _panel


def source_tensor_grid(f, x, y, z):
    """Return a source tensor-grid value, or NotImplemented for fallback."""
    shape = jnp.shape(x)
    if len(shape) != 3 or 0 in shape or jnp.shape(y) != shape or jnp.shape(z) != shape:
        return NotImplemented
    # Recognition is eager. Retain existing JIT/AD behavior for every tracer,
    # including differentiation of factor/core coefficients with fixed points.
    if any(isinstance(v, jax.core.Tracer) for v in jax.tree.leaves((f, x, y, z))):
        return NotImplemented
    # Preserve the established Python real-coordinate conversion.
    points = tuple(jnp.asarray(v, dtype=jnp.float64) for v in (x, y, z))
    patterns = (((0, 1, 2), (0, 1, 2)), ((2, 0, 1), (0, 1, 2)),
                ((1, 2, 0), (0, 1, 2)), ((1, 0, 2), (1, 0, 2)))
    for axes, order in patterns:
        vectors = []
        for point, axis in zip(points, axes):
            index = tuple(slice(None) if k == axis else 0 for k in range(3))
            vector = point[index]
            view_shape = tuple(vector.size if k == axis else 1 for k in range(3))
            if not bool(jnp.max(jnp.abs(point-vector.reshape(view_shape))) == 0):
                break
            vectors.append(vector)
        else:
            factors = (f.cols, f.rows, f.tubes)
            values = []
            # Native evaluates columns, rows and tubes before contractions.
            for k, vector in enumerate(vectors):
                panel = _panel(factors[k], tuple(f.domain[2*k:2*k+2]))
                values.append(jnp.asarray(panel(vector)).reshape((vector.size, -1)))
            permutation = tuple(axes.index(k) for k in range(3))
            result = jnp.transpose(f.core, permutation)
            for k in order:
                result = f.txm(result, values[k], axes[k]+1)
            return result
    return NotImplemented
