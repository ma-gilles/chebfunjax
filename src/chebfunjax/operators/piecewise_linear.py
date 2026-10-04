"""Source-shaped projected matrix assembly for scalar piecewise BVPs.

This is a bounded assembly primitive, not a public solve path. It constructs
the MATLAB order-two scalar collocation system from panel-local coefficient
fields and canonical second-kind differentiation matrices.

Provenance
----------
MATLAB source: ``@chebcolloc/reduce.m``, ``@opDiscretization/matrix.m``,
``@opDiscretization/getConstraints.m``, ``@opDiscretization/partition.m``,
``@valsDiscretization/{points,rhs}.m``, ``@chebcolloc2/{functionPoints,
equationPoints,toFunctionOut}.m``, and ``@linop/{deriveContinuity,
linsolve}.m``.
Chebfun commit: ``7574c77``. Original authors: Copyright 2017 by
The University of Oxford and the Chebfun Developers.
The pinned ``@linop/deriveContinuity.m`` source has SHA-256
``d8c37f431f148af4c3e9ca774d54a1b222e53939c6ee69a4499a5e8905399914``;
its derivative-order-then-breakpoint loops determine the grouped constraint
row order used here. Physical panel coordinates use the operation order in
``@valsDiscretization/points.m`` (SHA-256
``f879c65682d77ddb84f1246f14855350432bbd308d571a29a8c644110e438491``).

The unknown vector consists of Chebtech2 values, with ``d + 2`` values on a
panel of base dimension ``d``. Differential equations are projected to the
``d`` Chebtech1 equation points. Continuity and endpoint constraints precede
the projected equation rows. This helper does not implement coefficient
extraction, boundary-condition discovery, adaptive dimensions, or solve
dispatch; the caller supplies canonical derivative matrices and effective
linear right-hand-side fields.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.numpy as jnp

from chebfunjax.utils.diffmat import _cheb1_angles, _cheb2_angles, _cheb2_barywts
from chebfunjax.utils.interpolation import barymat
from chebfunjax.utils.quadrature import chebpts

ScalarField = Callable[[jnp.ndarray], jnp.ndarray] | float | complex


def _sample_field(field: ScalarField, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate and shape a scalar coefficient or panel-local field."""
    values = jnp.asarray(field(x) if callable(field) else field)
    if values.ndim == 0:
        return jnp.broadcast_to(values, x.shape)
    if values.size != x.size:
        raise ValueError("a panel field must return one value per node")
    return jnp.reshape(values, x.shape)


def _map_reference_nodes(a: jnp.ndarray, b: jnp.ndarray, reference: jnp.ndarray):
    """Map nodes in literal valsDiscretization.points multiplication order."""
    return b * (reference + 1.0) / 2.0 + a * (1.0 - reference) / 2.0


def _panel_projection(
    a: jnp.ndarray, b: jnp.ndarray, base_n: int, raw_n: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the Chebtech2-to-Chebtech1 projection and physical nodes."""
    eq_ref = chebpts(base_n, kind=1)
    raw_ref = chebpts(raw_n, kind=2)
    theta_eq = _cheb1_angles(base_n)
    theta_raw = _cheb2_angles(raw_n)
    eq_x = _map_reference_nodes(a, b, eq_ref)
    raw_x = _map_reference_nodes(a, b, raw_ref)
    projection = barymat(
        eq_x,
        raw_x,
        _cheb2_barywts(raw_n),
        theta_eq,
        theta_raw,
        do_flip=True,
    )
    return projection, eq_x, raw_x


def _build_scalar_projected_system(
    physical_domains: jnp.ndarray,
    base_dimensions: tuple[int, ...],
    coefficient_fields_by_panel: Sequence[Sequence[ScalarField]],
    canonical_raw_d_by_panel: Sequence[Sequence[jnp.ndarray]],
    rhs_fields_by_panel: Sequence[ScalarField] | None = None,
    continuity_values: jnp.ndarray | None = None,
    endpoint_values: tuple[complex | float, complex | float] = (0.0, 0.0),
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build a scalar, linear, order-two projected collocation system.

    ``base_dimensions`` and the panel count are static metadata. Every panel
    requires coefficient fields ``(c0, c1, c2)`` and canonical raw derivative
    matrices ``(D0, D1, D2)`` of size ``(d+2, d+2)``. Derivative matrices are
    with respect to the canonical coordinate on ``[-1, 1]``; physical
    scaling is applied here. The RHS fields are sampled directly at
    first-kind equation nodes and are already the effective linear RHS.

    Returns ``(A, b)`` with constraints first: all value-continuity rows in
    breakpoint order, then all derivative-continuity rows in breakpoint
    order, then the two Dirichlet endpoint conditions, followed by projected
    equation rows panel by panel. The grouped continuity order follows
    ``@linop/deriveContinuity.m``.

    Source adapters: only scalar second-order operators with C1 interfaces
    and endpoint-value constraints are represented. General boundary
    functionals, jumps, systems, adaptive dimension selection, and affine
    RHS extraction remain outside this helper. All numerical operations use
    JAX; Python loops are over static panel and derivative metadata.
    """
    panel_count = len(base_dimensions)
    if panel_count == 0:
        raise ValueError("at least one panel is required")
    if len(coefficient_fields_by_panel) != panel_count:
        raise ValueError("coefficient fields must be supplied for every panel")
    if len(canonical_raw_d_by_panel) != panel_count:
        raise ValueError("raw derivative matrices must be supplied per panel")
    if rhs_fields_by_panel is not None and len(rhs_fields_by_panel) != panel_count:
        raise ValueError("RHS fields must be supplied for every panel")
    if any(len(fields) != 3 for fields in coefficient_fields_by_panel):
        raise ValueError("an order-two scalar operator requires c0, c1, c2")
    if any(len(mats) != 3 for mats in canonical_raw_d_by_panel):
        raise ValueError("an order-two scalar operator requires D0, D1, D2")
    if any(not isinstance(n, int) or n < 1 for n in base_dimensions):
        raise ValueError("base dimensions must be positive static integers")

    domains = jnp.asarray(physical_domains)
    if domains.ndim != 1 or domains.shape != (panel_count + 1,):
        raise ValueError("physical_domains must have one endpoint per panel plus one")

    raw_dimensions = tuple(n + 2 for n in base_dimensions)
    unknown_offsets = [0]
    equation_offsets = [0]
    for d, q in zip(base_dimensions, raw_dimensions):
        unknown_offsets.append(unknown_offsets[-1] + q)
        equation_offsets.append(equation_offsets[-1] + d)
    unknown_count = unknown_offsets[-1]
    equation_count = equation_offsets[-1]
    constraint_count = 2 * panel_count

    projections = []
    scaled_derivatives = []
    rhs_samples = []
    sampled_coefficients = []
    for p, (d, q) in enumerate(zip(base_dimensions, raw_dimensions)):
        a, b = domains[p], domains[p + 1]
        projection, x_eq, x_raw = _panel_projection(a, b, d, q)
        projections.append(projection)
        derivatives = tuple(
            jnp.asarray(mat) * ((2.0 / (b - a)) ** k)
            for k, mat in enumerate(canonical_raw_d_by_panel[p])
        )
        if any(mat.shape != (q, q) for mat in derivatives):
            raise ValueError(f"panel {p}: each canonical D_k must have shape ({q}, {q})")
        scaled_derivatives.append(derivatives)
        sampled_coefficients.append(tuple(
            _sample_field(field, x_raw)
            for field in coefficient_fields_by_panel[p]
        ))
        rhs_samples.append(
            jnp.zeros((d,), dtype=domains.dtype)
            if rhs_fields_by_panel is None
            else _sample_field(rhs_fields_by_panel[p], x_eq)
        )

    dtype_args = [domains]
    dtype_args.extend(v for panel in sampled_coefficients for v in panel)
    dtype_args.extend(rhs_samples)
    dtype_args.extend(m for panel in scaled_derivatives for m in panel)
    dtype_args.append(jnp.asarray(endpoint_values))
    if continuity_values is not None:
        dtype_args.append(jnp.asarray(continuity_values))
    dtype = jnp.result_type(*dtype_args)

    # Constraint RHS uses the same derivative-major order as deriveContinuity:
    # values at all interfaces, followed by first derivatives at all interfaces.
    continuity_rhs = (
        jnp.zeros((2 * (panel_count - 1),), dtype=dtype)
        if continuity_values is None
        else jnp.reshape(jnp.asarray(continuity_values, dtype=dtype),
                         (2 * (panel_count - 1),))
    )
    constraint_rhs = jnp.concatenate((
        continuity_rhs,
        jnp.asarray(endpoint_values, dtype=dtype).reshape((2,)),
    ))
    constraint_rows = jnp.zeros((constraint_count, unknown_count), dtype=dtype)

    # MATLAB deriveContinuity loops derivative order outside breakpoint.
    # Preserve that grouped ordering for three or more panels.
    row = 0
    for p in range(panel_count - 1):
        left_start = unknown_offsets[p]
        right_start = unknown_offsets[p + 1]
        left_q, right_q = raw_dimensions[p], raw_dimensions[p + 1]
        constraint_rows = constraint_rows.at[row, left_start + left_q - 1].set(1)
        constraint_rows = constraint_rows.at[row, right_start].add(-1)
        row += 1
    for p in range(panel_count - 1):
        left_start = unknown_offsets[p]
        right_start = unknown_offsets[p + 1]
        left_q, right_q = raw_dimensions[p], raw_dimensions[p + 1]
        constraint_rows = constraint_rows.at[
            row, left_start:left_start + left_q
        ].set(scaled_derivatives[p][1][-1, :])
        constraint_rows = constraint_rows.at[
            row, right_start:right_start + right_q
        ].add(-scaled_derivatives[p + 1][1][0, :])
        row += 1

    constraint_rows = constraint_rows.at[row, unknown_offsets[0]].set(1)
    constraint_rows = constraint_rows.at[row + 1, unknown_offsets[-1] - 1].set(1)

    operator_rows = jnp.zeros((equation_count, unknown_count), dtype=dtype)
    for p, (d, q) in enumerate(zip(base_dimensions, raw_dimensions)):
        raw_block = jnp.zeros((q, q), dtype=dtype)
        for k in range(3):
            raw_block = raw_block + (
                sampled_coefficients[p][k][:, None] * scaled_derivatives[p][k]
            )
        projected_block = projections[p] @ raw_block
        r0, r1 = equation_offsets[p], equation_offsets[p + 1]
        c0, c1 = unknown_offsets[p], unknown_offsets[p + 1]
        operator_rows = operator_rows.at[r0:r1, c0:c1].set(projected_block)

    return (
        jnp.concatenate((constraint_rows, operator_rows), axis=0),
        jnp.concatenate((constraint_rhs, *rhs_samples)),
    )


def _project_scalar_raw_solution(
    physical_domains: jnp.ndarray,
    base_dimensions: tuple[int, ...],
    raw_solution: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    """Project concatenated raw panel values to source first-kind samples.

    The returned tuple contains one ``base_dimensions[p]`` vector per panel.
    MATLAB ``linsolve`` applies this projection before partitioning and output
    conversion. Shape validation prevents silently ignoring trailing values.
    """
    panel_count = len(base_dimensions)
    if panel_count == 0:
        raise ValueError("at least one panel is required")
    if any(not isinstance(n, int) or n < 1 for n in base_dimensions):
        raise ValueError("base dimensions must be positive static integers")
    domains = jnp.asarray(physical_domains)
    values = jnp.asarray(raw_solution)
    if domains.ndim != 1 or domains.shape != (panel_count + 1,):
        raise ValueError("physical_domains must have one endpoint per panel plus one")
    raw_dimensions = tuple(n + 2 for n in base_dimensions)
    expected_size = sum(raw_dimensions)
    if values.ndim != 1 or values.shape != (expected_size,):
        raise ValueError(f"raw_solution must be a vector of length {expected_size}")
    out = []
    offset = 0
    for p, (d, q) in enumerate(zip(base_dimensions, raw_dimensions)):
        projection, _eq_nodes, _raw_nodes = _panel_projection(
            domains[p], domains[p + 1], d, q
        )
        out.append(projection @ values[offset:offset + q])
        offset += q
    return tuple(out)
