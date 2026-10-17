"""Real binary64 block 1-norm estimation for native full-mass initialization.

Provenance
----------
MATLAB source: R2025b normest1.m, including undupli and mysign; condest.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
Algorithm: Higham and Tisseur, SIAM J. Matrix Anal. Appl.21 (2000),1185–1201.
Numerical actions and random draws use JAX; adaptive control is eager.
"""

import jax
import jax.numpy as jnp

from ._matlab_uniform_stream import default_uniform_stream
from ._normest_triangular import triangular_solve
from ._source_lu_factor import general_factor


@jax.jit
def _condition_lu(a):
    """Use the source real LU schedule before estimating inverse actions.

    Provenance: native dense MATLAB LU; existing _source_lu_factor controls.
    The default CPU provider has different rounding on ill-conditioned inputs.
    """
    return general_factor(a)[0]


def _sign(a):
    return jnp.where(a == 0, 1.0, jnp.sign(a))


def _undupli(signs, old, stream):
    n, t = signs.shape
    if t == 1:
        return signs, 0
    if old is None:
        previous = signs[:, :1]
        start = 1
    else:
        previous = old
        start = 0
    repeated = 0
    for column in range(start, t):
        attempts = 0
        while float(jnp.max(jnp.abs(signs[:, column] @ previous))) == n:
            attempts += 1
            fresh = _sign(2 * stream.uniform((n, 1)) - jnp.ones((n, 1)))
            signs = signs.at[:, column].set(fresh[:, 0])
            if attempts > n / t:
                break
        repeated += int(attempts > 0)
        if column < t - 1:
            previous = jnp.concatenate((previous, signs[:, column : column + 1]), axis=1)
    return signs, repeated


def normest1_real(action, transpose_action, n, *, stream=None, t=2):
    """Source block estimator for a real square operator and persistent RNG.

    Return its estimate, maximizing basis vector/image, iteration diagnostics,
    and termination reason. Only the real binary64 operator route is exposed.

    Provenance: MATLAB R2025b normest1.m; Chebfun7574c77 pde15s consumer.
    """
    stream = default_uniform_stream if stream is None else stream
    if not isinstance(n, int) or n < 1:
        raise ValueError("Operator dimension must be a positive integer")
    if not isinstance(t, int) or not 1 <= t <= max(n, 2):
        raise ValueError("Block size outside source range")
    if t == n or n <= 4:
        y = action(jnp.eye(n, dtype=jnp.float64))
        norms = jnp.sum(jnp.abs(y), axis=0)
        column = int(jnp.argsort(norms, stable=True)[-1])
        return dict(
            estimate=float(norms[column]),
            v=jnp.eye(n, dtype=jnp.float64)[:, column],
            w=y[:, column],
            iterations=(0, 1),
            repetitions=(0, 0),
            reason=0,
        )

    x = jnp.ones((n, t), dtype=jnp.float64)
    if t > 1:
        x = x.at[:, 1:].set(_sign(2 * stream.uniform((n, t - 1)) - 1))
    x, _ = _undupli(x, None, stream)
    x = x / n
    indices = jnp.zeros(t, dtype=jnp.int32)
    signs = jnp.zeros((n, t), dtype=jnp.float64)
    estimate_old = 0.0
    iteration = products = repeated_signs = repeated_units = 0
    index_history = []
    while True:
        iteration += 1
        y = action(x)
        products += 1
        norms = jnp.sum(jnp.abs(y), axis=0)
        ranked = jnp.argsort(norms, stable=True)[::-1]
        best = int(ranked[0])
        estimate = float(norms[best])
        if estimate > estimate_old or iteration == 2:
            estimate_column = int(indices[best])
            w = y[:, best]
        if iteration >= 2 and estimate <= estimate_old:
            estimate = estimate_old
            reason = 2
            break
        estimate_old = estimate
        if iteration > 5:
            iteration = 5
            reason = 1
            break
        old = signs
        signs = _sign(y)
        parallels = jnp.max(jnp.abs(old.T @ signs), axis=0) == n
        if int(jnp.sum(parallels)) == t:
            reason = 3
            break
        signs, repetitions = _undupli(signs, old, stream)
        repeated_signs += repetitions
        z = transpose_action(signs)
        products += 1
        scores = jnp.max(jnp.abs(z), axis=1)
        if iteration >= 2 and float(jnp.max(scores)) == float(scores[estimate_column]):
            reason = 4
            break
        ranked = jnp.argsort(scores, stable=True)[::-1]
        count = t
        if iteration == 1:
            indices = ranked[:t]
            index_history = [int(i) for i in indices]
        else:
            repeats = sum(int(i) in index_history for i in ranked[:t])
            repeated_units += repeats
            if repeats == t:
                reason = 5
                break
            fresh = [int(i) for i in ranked if int(i) not in index_history][:t]
            count = len(fresh)
            indices = indices.at[:count].set(jnp.asarray(fresh, dtype=jnp.int32))
            index_history.extend(fresh)
        x = jnp.zeros((n, t), dtype=jnp.float64)
        for column in range(count):
            x = x.at[indices[column], column].set(1.0)
    v = jnp.zeros(n, dtype=jnp.float64).at[estimate_column].set(1.0)
    return dict(
        estimate=estimate,
        v=v,
        w=w,
        iterations=(iteration, products),
        repetitions=(repeated_signs, repeated_units),
        reason=reason,
    )


def condest_real(a, *, stream=None):
    """Native real dense LU-based condition estimate, with source random signs.

    The triangular operator intentionally omits the LU row permutation, as
    R2025b condestf does. Caller supplies an explicit stream for native replay;
    default calls advance the shared persistent uniform stream.

    Provenance: MATLAB R2025b condest.m/normest1.m; Chebfun7574c77 pde15s.
    """
    a = jnp.asarray(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Condition estimate requires a square matrix")
    if a.dtype != jnp.float64:
        raise NotImplementedError("Condition estimator currently requires real binary64")
    n = a.shape[0]
    if n == 0:
        return 0.0
    packed = _condition_lu(a)
    upper = jnp.triu(packed)
    lower = jnp.tril(packed, -1) + jnp.eye(n, dtype=a.dtype)
    if bool(jnp.any(jnp.diag(upper) == 0)):
        return float("inf")

    def action(x):
        return triangular_solve(
            upper, triangular_solve(lower, x, lower=True, unit=True), lower=False
        )

    def transpose(x):
        return triangular_solve(
            lower,
            triangular_solve(upper, x, lower=False, transpose=True),
            lower=True,
            unit=True,
            transpose=True,
        )

    estimated = normest1_real(action, transpose, n, stream=stream)
    return estimated["estimate"] * float(jnp.max(jnp.sum(jnp.abs(a), axis=0)))
