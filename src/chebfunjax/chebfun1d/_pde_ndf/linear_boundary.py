"""Linear endpoint rows for polynomial pdeSolve boundary shorthand.

Provenance
----------
MATLAB @chebfun/pdeSolve.m: neumannLeft/neumannRight and endpoint parsing.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
The source multiplies canonical Neumann rows by domain width/2.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class _LinearEndpoint:
    kind: str
    side: int
    component: int
    target: object = 0.0


def _boundary_token(value):
    if not isinstance(value, str):
        raise TypeError("Boundary shorthand must be a string")
    token = value.lower()
    if token not in ("neumann", "dirichlet"):
        raise NotImplementedError("Polynomial boundary shorthand requires neumann or dirichlet")
    return token


def _split_endpoint_specs(lbc, rbc, size, system):
    linear, nonlinear = [], []
    for side, specs in ((0, lbc), (-1, rbc)):
        if specs is None:
            continue
        for spec in specs if isinstance(specs, (tuple, list)) else [specs]:
            if isinstance(spec, str):
                token = _boundary_token(spec)
                linear.extend(_LinearEndpoint(token, side, k) for k in range(size))
            elif callable(spec):
                nonlinear.append((side, spec))
            elif system:
                raise NotImplementedError("Numeric system boundary targets are not implemented")
            else:
                linear.append(_LinearEndpoint("dirichlet", side, 0, spec))
    return linear, nonlinear


def _endpoint_sum(row):
    # Native n=9/33 captures qualify index-order accumulation here.
    # A generic reduction changes the n=33 left diagonal by one ulp.
    return jax.lax.fori_loop(
        0, row.size, lambda index, total: total + row[index], jnp.zeros((), row.dtype)
    )


def _canonical_endpoint_row(n, side):
    # Preserve the literal source cosine, reciprocal, sign, diagonal schedule.
    x = -jnp.cos(jnp.pi * jnp.arange(1, n - 1, dtype=jnp.float64) / (n - 1))
    if side == 0:
        row = jnp.concatenate((jnp.zeros(1), 2 / (1 + x), jnp.asarray([0.5])))
        row = row.at[::2].multiply(-1)
        return row.at[0].set(-_endpoint_sum(row))
    row = jnp.concatenate((jnp.asarray([-0.5]), 2 / (x - 1), jnp.zeros(1)))
    row = row.at[jnp.arange(n - 1, -1, -2)].multiply(-1)
    return row.at[-1].set(-_endpoint_sum(row))


def _linear_boundary_matrix(descriptors, n, size, domain, dtype):
    matrix = jnp.zeros((len(descriptors), n * size), dtype=dtype)
    for index, spec in enumerate(descriptors):
        if spec.kind == "neumann":
            row = _canonical_endpoint_row(n, spec.side) * ((domain[1] - domain[0]) / 2)
        else:
            row = jnp.zeros(n, dtype=dtype).at[spec.side].set(1)
        start = spec.component * n
        matrix = matrix.at[index, start : start + n].set(row)
    targets = jnp.asarray([spec.target for spec in descriptors], dtype=dtype)
    return matrix, targets
