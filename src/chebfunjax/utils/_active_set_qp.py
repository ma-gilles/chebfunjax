"""Private native active-set QP primitives; not an optimizer.

Provenance: MATLAB R2017a shared/optimlib/compdir.m (including choltrap),
qpsub.m977-1007. Source hashes: native_box_active_set_source_20261009/
SOURCE_BINDINGS_v1.json. Caller target: Chebfun 7574c77680d7e82b79626300bf255498271a72df.

Real binary64, ambient dimension two, projected dimension one or two.
Finite inputs are the intended contract; no MATLAB nonfinite or bit parity
claim. No finite differences, objective callback, fallback or public dispatch.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

NEWTON, NEGATIVE_CURVATURE, STEEPEST_DESCENT = 0, 1, 2


class Direction(NamedTuple):
    step: jax.Array
    kind: jax.Array


class Blocking(NamedTuple):
    eligible: jax.Array  # Source indf encoded as a fixed four-entry boolean mask.
    index: jax.Array  # Zero-based original constraint index; -1 means source [].
    distance: jax.Array


def _real64(value, name):
    result = jnp.asarray(value)
    if result.dtype != jnp.float64:
        raise TypeError(f'{name} requires real binary64 inputs')
    return result


def choltrap(matrix):
    """Literal local choltrap for 1x1/2x2; fixed-shape absent direction mask.

    Return (L, sneg, present); sneg is zero-filled when source returns [].
    The source updates the LOWER triangle; do not symmetrize that update.
    """
    a = _real64(matrix, 'matrix')
    if a.shape not in ((1, 1), (2, 2)):
        raise ValueError('choltrap supports projected dimensions one and two')
    n = a.shape[0]
    identity = jnp.eye(n, dtype=a.dtype)
    zero = jnp.zeros(n, dtype=a.dtype)
    if n == 1:
        return jax.lax.cond(
            a[0, 0] <= 0,
            lambda: (identity, identity[:, 0], jnp.asarray(True)),
            lambda: (identity.at[0, 0].set(jnp.sqrt(a[0, 0])), zero,
                     jnp.asarray(False)))

    def positive_first():
        ell = identity.at[0, 0].set(jnp.sqrt(a[0, 0]))
        ell = ell.at[1, 0].set(a[1, 0] / ell[0, 0])
        last = a[1, 1] - ell[1, 0] * ell[1, 0]
        return jax.lax.cond(
            last <= 0,
            lambda: (ell, solve_triangular(ell.T, identity[:, 1], lower=False),
                     jnp.asarray(True)),
            lambda: (ell.at[1, 1].set(jnp.sqrt(last)), zero, jnp.asarray(False)))

    return jax.lax.cond(
        a[0, 0] <= 0,
        lambda: (identity, identity[:, 0], jnp.asarray(True)), positive_first)


def _dot2(a, b):
    """Two-term source dot with each product rounded before accumulation."""
    return a[0]*b[0] + a[1]*b[1]


def _matvec2(matrix, vector):
    return jnp.stack((_dot2(matrix[0], vector), _dot2(matrix[1], vector)))


def _scalar_matrix_divide(matrix, denominator):
    """Retain scalar division instead of XLA broadcast reciprocal multiply.

    This helper is eager: fusing its scalar operations under jit can reintroduce
    the rewrite. The fixed two-variable SQP driver uses host control throughout.
    Qualified against 14 saved native BFGS states; no compensated arithmetic.
    """
    return jnp.stack([matrix.reshape(-1)[i]/denominator
                      for i in range(matrix.size)]).reshape(matrix.shape)


def _scalar_newton_substitution(upper, gradient):
    """Ordinary forward/back substitution, with eager scalar divisions."""
    first = gradient[0]/upper[0, 0]
    if upper.shape == (1, 1):
        return jnp.stack((first/upper[0, 0],))
    second = (gradient[1]-upper[0, 1]*first)/upper[1, 1]
    last = second/upper[1, 1]
    initial = (first-upper[0, 1]*last)/upper[0, 0]
    return jnp.stack((initial, last))


def compdir(z, hessian, gradient, *, scalar_arithmetic=False):
    """Native projected direction and kind; Z=I represents source scalar Z=1.

    Z must have shape (2,1) or (2,2). Native chol reads the upper triangle;
    transpose before JAX lower Cholesky, without input symmetrization.
    """
    z = _real64(z, 'z')
    h = _real64(hessian, 'hessian')
    g = _real64(gradient, 'gradient')
    if z.shape not in ((2, 1), (2, 2)) or h.shape != (2, 2) or g.shape != (2,):
        raise ValueError('compdir requires ambient dimension two and projection one or two')
    projected = (z.T @ h) @ z
    upper = jax.lax.linalg.cholesky(projected.T, symmetrize_input=False).T
    positive = jnp.all(jnp.isfinite(jnp.diag(upper))) & jnp.all(jnp.diag(upper) > 0)

    # nlconst/qpsub are host-controlled. Keep the eager source arithmetic
    # separate from the existing traceable primitive used by direct JIT callers.
    # XLA's matrix division rewrite and blocked triangular solve differ in
    # rounding from the native backend on the captured two-variable states.
    if scalar_arithmetic and bool(positive):
        step = -z @ _scalar_newton_substitution(upper, z.T @ g)
        if bool(g @ step > 0):
            step = -step
        return Direction(step, jnp.asarray(NEWTON))

    def newton():
        step = -z @ solve_triangular(
            upper, solve_triangular(upper.T, z.T @ g, lower=True), lower=False)
        return step, jnp.asarray(NEWTON)

    def nonpositive():
        _, negative, present = choltrap(projected)
        sufficient = present & (((negative @ projected) @ negative)
                                < -jnp.sqrt(jnp.finfo(jnp.float64).eps))
        return jax.lax.cond(
            sufficient,
            lambda: (z @ negative, jnp.asarray(NEGATIVE_CURVATURE)),
            lambda: (-z @ (z.T @ g), jnp.asarray(STEEPEST_DESCENT)))

    step, kind = jax.lax.cond(positive, newton, nonpositive)
    step = jax.lax.cond(g @ step > 0, lambda: -step, lambda: step)
    return Direction(step, kind)


def find_blocking_constraint(a, step, residual, errnorm, active):
    """qpsub.findBlockingConstr for four inequalities in two variables.

    Includes general four-row A fixtures; the future native box caller uses
    [-I;I]. Fixed mask/-1 encode source variable-length indices and empty [].
    """
    a = _real64(a, 'a')
    step = _real64(step, 'step')
    residual = _real64(residual, 'residual')
    errnorm = _real64(errnorm, 'errnorm')
    active = jnp.asarray(active)
    if (a.shape != (4, 2) or step.shape != (2,) or residual.shape != (4,)
            or errnorm.shape != () or active.shape != (4,)):
        raise ValueError('blocking constraint requires four inequalities in dimension two')
    if active.dtype != jnp.bool_:
        raise TypeError('active requires a boolean mask')
    directional = a @ step
    eligible = (directional > errnorm * jnp.linalg.norm(step)) & ~active
    # Only eligible entries are divided in native variable-length indexing.
    distances = jnp.where(eligible, jnp.abs(residual) / jnp.where(eligible, directional, 1),
                          jnp.inf)
    nearest = jnp.min(distances)
    first = jnp.argmax(eligible & (distances == nearest))
    any_eligible = jnp.any(eligible)
    return Blocking(eligible, jnp.where(any_eligible, first, -1),
                    jnp.where(any_eligible, nearest, 1e16))
