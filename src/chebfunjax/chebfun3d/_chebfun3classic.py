"""Adaptive MACA / block-term classic Chebfun3 construction.

Provenance: @chebfun3/chebfun3classic.m and private/sampleTest.m,
Chebfun 7574c77680d7e82b79626300bf255498271a72df.
Copyright The University of Oxford and The Chebfun Developers.
JAX numerical arrays; eager rank decisions and variable-size skeleton lists.
"""

import math
import warnings
from dataclasses import dataclass

import jax.numpy as jnp

from chebfunjax.chebfun3d._chebfun3double import (
    _pivot2,
    _pivot3,
    _subtract2,
    _subtract3,
    btd_to_tucker,
    tensor_axes,
    tensor_tolerance,
)
from chebfunjax.chebfun3d._chebfun3f_source import unfold
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech


@dataclass
class Skeleton:
    blocks: list
    indices2: list
    fibers: object
    pivots: object
    indices3: object
    fail3: bool
    fail2: bool
    positions: object = None


def aca2(values, tolerance, factor):
    """Source completeACA2D, including factor stopping and failure flag."""
    matrix = jnp.asarray(values)
    nx, ny = matrix.shape
    width = min(nx, ny)
    limit = width / factor if factor else math.inf
    row, col, magnitude = _pivot2(matrix, tolerance)
    if bool(magnitude == 0):
        return (
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            jnp.zeros((1, 1)),
            jnp.zeros((1, 2), dtype=jnp.int64),
            False,
        )
    columns, rows, pivots, locations = [], [], [], []
    while bool(magnitude > tolerance) and len(pivots) < limit and len(pivots) < width:
        columns.append(matrix[:, col])
        rows.append(matrix[row, :])
        pivots.append(matrix[row, col])
        locations.append(jnp.stack((row, col)))
        matrix = _subtract2(matrix, row, col)
        row, col, magnitude = _pivot2(matrix, tolerance)
    failed = not bool(magnitude <= tolerance) or len(pivots) >= limit
    if not pivots:
        return (
            jnp.zeros((nx, 1)),
            jnp.zeros(1),
            jnp.zeros((ny, 1)),
            jnp.zeros((1, 2), dtype=jnp.int64),
            failed,
        )
    return (
        jnp.stack(columns, axis=1),
        jnp.stack(pivots),
        jnp.stack(rows, axis=1),
        jnp.stack(locations),
        failed,
    )


def aca3(values, axes, domain, epsilon, tolerance, factor=2 * math.sqrt(2)):
    """Source completeACA3D: residual slices, two pivot lists and failure flags."""
    tensor = jnp.asarray(values)
    n1, n2, n3 = tensor.shape
    width = min(n3, n1 * n2)
    limit = width / factor if factor else math.inf
    first, second, third, magnitude = _pivot3(tensor)
    blocks, indices2, fibers, pivots, locations = [], [], [], [], []
    fail2 = True
    if bool(magnitude == 0):
        return Skeleton(
            [(jnp.zeros((1, 1)), jnp.zeros(1), jnp.zeros((1, 1)))],
            [jnp.zeros((1, 2), dtype=jnp.int64)],
            jnp.zeros((1, 1)),
            jnp.zeros(1),
            jnp.zeros((1, 3), dtype=jnp.int64),
            False,
            False,
        )
    count = 0
    while bool(magnitude > tolerance) and count < limit and count < width:
        fiber = tensor[first, second, :]
        matrix = tensor[:, :, third]
        pivot = tensor[first, second, third]
        if count == 0:
            _, slice_tolerance = tensor_tolerance(matrix, axes[:2], domain[:4], epsilon)
        columns, p2, rows, loc2, failed = aca2(matrix, slice_tolerance, 2 if factor else 0)
        if count == 0:
            fail2 = failed
        blocks.append((columns, p2, rows))
        indices2.append(loc2)
        fibers.append(fiber)
        pivots.append(pivot)
        locations.append(jnp.stack((first, second, third)))
        tensor = _subtract3(tensor, columns, p2, rows, fiber, pivot)
        count += 1
        first, second, third, magnitude = _pivot3(tensor)
    if count and bool(jnp.all(blocks[-1][1] == 0)):
        blocks.pop()
        indices2.pop()
        fibers.pop()
        pivots.pop()
        locations.pop()
        magnitude = 0
    fail3 = not bool(magnitude <= tolerance)
    if count == 0 and not fail3:
        fail2 = False
    if count > limit:
        fail3 = True
    if count == 0:
        return Skeleton(
            [(jnp.zeros((n1, 1)), jnp.zeros(1), jnp.zeros((n2, 1)))],
            [jnp.zeros((1, 2), dtype=jnp.int64)],
            jnp.zeros((n3, 1)),
            jnp.zeros(1),
            jnp.zeros((1, 3), dtype=jnp.int64),
            fail3,
            fail2,
        )
    return Skeleton(
        blocks,
        indices2,
        jnp.stack(fibers, axis=1) if fibers else jnp.zeros((n3, 0)),
        jnp.stack(pivots) if pivots else jnp.zeros(0),
        jnp.stack(locations) if locations else jnp.zeros((0, 3), dtype=jnp.int64),
        fail3,
        fail2,
    )


def tolerance3(values, axes, domain, epsilon, grid):
    """Classic uses explicit grid, even when grid2D has refined further."""
    relative = 2 * jnp.asarray(grid, dtype=jnp.float64) ** (4 / 5) * epsilon
    gradients = []
    for axis in range(3):
        selection = tuple(
            slice(None) if k == axis else slice(0, n - 1) for k, n in enumerate(values.shape)
        )
        difference = jnp.diff(values[selection], axis=axis)
        if difference.size:
            shape = [1] * 3
            shape[axis] = values.shape[axis] - 1
            gradients.append(jnp.max(jnp.abs(difference / jnp.diff(axes[axis]).reshape(shape))))
    gradients = jnp.stack(gradients) if gradients else jnp.asarray(1.0)
    lengths = jnp.asarray(domain[1::2]) - jnp.asarray(domain[::2])
    scale = jnp.maximum(jnp.max(gradients * lengths), jnp.max(jnp.abs(values)))
    return float(relative), scale * relative


def refine1(grid, tech):
    grid = math.floor(math.sqrt(2) ** (math.floor(2 * math.log2(grid)) + 1)) + 1
    return math.floor(grid / 2 + 0.5) * 2 if tech is Trigtech else grid


def refine2(grid, tech):
    if tech is Chebtech1:
        return 3 * grid, 3, 1
    return (2 * grid if tech is Trigtech else 2 * grid - 1), 2, 0


def evaluate(op, coordinates, vectorize):
    coordinates = jnp.broadcast_arrays(*coordinates)
    if vectorize:
        shape = coordinates[0].shape
        flat = [x.reshape(-1, order="F") for x in coordinates]
        result = jnp.asarray([op(*point) for point in zip(*flat)])
        return result.reshape(shape, order="F")
    result = jnp.asarray(op(*coordinates))
    # Native only transposes mismatched output dimensions; its vectorCheck
    # adapts scalar callbacks before entering this constructor.
    if result.ndim == 0:
        result = jnp.broadcast_to(result, coordinates[0].shape)
    if result.shape != coordinates[0].shape:
        result = result.T
    return result


def vector_check(op, domain):
    """Classic-only adapter of constructor.m's common vectorCheck parser step.

    The first endpoint call is protected; the second is not. Keep both calls,
    including the native scalar-return wrapping and nonfinite error order.
    """
    axes = [jnp.asarray(domain[k : k + 2]) for k in (0, 2, 4)]
    coordinates = jnp.meshgrid(*axes, indexing="ij")
    try:
        op(*coordinates)
    except Exception:
        warnings.warn(
            "CHEBFUN:CHEBFUN3:constructor:vectorize: Function did not correctly "
            "evaluate on an array. Turning on the 'vectorize' flag.",
            RuntimeWarning,
        )
        return op, True
    values = jnp.asarray(op(*coordinates))
    if bool(jnp.any(jnp.isinf(values))):
        raise ValueError("CHEBFUN:CHEBFUN3:constructor:inf: Function returned INF when evaluated")
    if bool(jnp.any(jnp.isnan(values))):
        raise ValueError("CHEBFUN:CHEBFUN3:constructor:nan: Function returned NaN when evaluated")
    if values.size == 1:
        original = op

        def op(x, y, z):
            return original(x, y, z) + 0 * x + 0 * y + 0 * z

    return op, False


def _on_axes(op, axes, vectorize):
    return evaluate(op, jnp.meshgrid(*axes, indexing="ij"), vectorize)


def _rank(matrix):
    singular = jnp.linalg.svd(matrix, compute_uv=False)
    return int(jnp.sum(singular > max(matrix.shape) * jnp.spacing(singular[0])))


def dim_cluster(op, domain, vectorize, tech):
    from chebfunjax.chebfun1d.chebfun import chebfun

    values = _on_axes(op, tensor_axes((10, 10, 10), domain, tech), vectorize)
    unfolded = [unfold(values, axis) for axis in range(3)]
    ranks = [_rank(matrix.T) for matrix in unfolded]
    indices = [axis for axis in range(3) if ranks[axis] == min(ranks)]
    if len(indices) == 1:
        return indices[0] + 1
    # Source intentionally constructs these numeric columns using session
    # defaults, not the selected classic preference/domain.
    lengths = [len(chebfun(matrix[:, 0]).simplify(1e-3)) for matrix in unfolded]
    return max(indices, key=lambda axis: lengths[axis]) + 1


def happy(values, tech, relative, pref, *, initial_vscale=None):
    from chebfunjax.chebfun3d._chebfun3f_tech import envelope_happiness

    return envelope_happiness(
        values, tech, relative, pref.happinessCheck, pref,
        initial_vscale=initial_vscale,
    )


def happiness_btd(skeleton, tech, relative, pref):
    """Initial BTD checks, returned in internal column/row/fiber order.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3classic.m, happinessCheckBTD
    Chebfun commit: 7574c77
    """
    columns, _, rows = skeleton.blocks[0]
    slice_scale = jnp.maximum(
        jnp.max(jnp.abs(columns)), jnp.max(jnp.abs(rows))
    )
    # Native evaluates fibers first. This order is observable by callbacks.
    resolved_fibers = happy(
        skeleton.fibers, tech, relative, pref,
        initial_vscale=jnp.max(jnp.abs(skeleton.fibers[:, 0])),
    )
    resolved_columns = happy(
        columns, tech, relative, pref, initial_vscale=slice_scale,
    )
    resolved_rows = happy(
        rows, tech, relative, pref, initial_vscale=slice_scale,
    )
    return [resolved_columns, resolved_rows, resolved_fibers]


def refine_skeleton(op, skeleton, axes, domain, tech, resolved, vectorize):
    """One literal Phase2 iteration: resample, then interleave 2D/3D ACA."""
    columns, rows = [b[0] for b in skeleton.blocks], [b[2] for b in skeleton.blocks]
    pivots2 = [b[1] for b in skeleton.blocks]
    diagonal = [jnp.diag(1 / p) for p in pivots2]
    if skeleton.positions is None:
        pos3 = [axes[k][skeleton.indices3[:, k]] for k in range(3)]
        pos2 = [[axes[k][indices[:, k]] for k in range(2)] for indices in skeleton.indices2]
    else:
        pos3, pos2 = skeleton.positions
    sizes = [axis.size for axis in axes]
    indices3, indices2 = skeleton.indices3, list(skeleton.indices2)
    for axis in range(3):
        if not resolved[axis]:
            sizes[axis], multiplier, offset = refine2(sizes[axis], tech)
            indices3 = indices3.at[:, axis].set(multiplier * indices3[:, axis] + offset)
            if axis < 2:
                indices2 = [p.at[:, axis].set(multiplier * p[:, axis] + offset) for p in indices2]
    refined_axes = tensor_axes(sizes, domain, tech)
    rank = skeleton.pivots.size
    fibers = jnp.column_stack(
        [
            _on_axes(
                op, [pos3[0][k : k + 1], pos3[1][k : k + 1], refined_axes[2]], vectorize
            ).reshape(-1)
            for k in range(rank)
        ]
    )
    if not resolved[0] or not resolved[1]:
        columns = [
            _on_axes(op, [refined_axes[0], pos2[k][1], pos3[2][k : k + 1]], vectorize)[:, :, 0]
            for k in range(rank)
        ]
        rows = [
            _on_axes(op, [pos2[k][0], refined_axes[1], pos3[2][k : k + 1]], vectorize)[:, :, 0].T
            for k in range(rank)
        ]
    for k in range(rank):
        if not resolved[0] or not resolved[1]:
            for i in range(pivots2[k].size - 1):
                columns[k] = (
                    columns[k]
                    .at[:, i + 1 :]
                    .add(
                        -(columns[k][:, i : i + 1] * diagonal[k][i, i])
                        @ rows[k][indices2[k][i + 1 :, 1], i : i + 1].T
                    )
                )
                rows[k] = (
                    rows[k]
                    .at[:, i + 1 :]
                    .add(
                        -(
                            (columns[k][indices2[k][i + 1 :, 0], i : i + 1] * diagonal[k][i, i])
                            @ rows[k][:, i : i + 1].T
                        ).T
                    )
                )
            for later in range(k + 1, rank):
                columns[later] = columns[later] - (
                    (columns[k] @ diagonal[k]) @ rows[k][indices2[later][:, 1], :].T
                ) * (fibers[indices3[later, 2], k] / skeleton.pivots[k])
            for later in range(k + 1, rank):
                rows[later] = rows[later] - (
                    (columns[k][indices2[later][:, 0], :] @ diagonal[k]) @ rows[k].T
                ).T * (fibers[indices3[later, 2], k] / skeleton.pivots[k])
        for later in range(k + 1, rank):
            scalar = (columns[k][indices3[later, 0], :] @ diagonal[k]) @ rows[k][
                indices3[later, 1], :
            ]
            fibers = fibers.at[:, later].add(-scalar * (fibers[:, k] / skeleton.pivots[k]))
    return Skeleton(
        list(zip(columns, pivots2, rows)),
        indices2,
        fibers,
        skeleton.pivots,
        indices3,
        skeleton.fail3,
        skeleton.fail2,
        (pos3, pos2),
    ), refined_axes


def _factors(matrix, interval, pref, relative, tech):
    """Public numeric construction, followed by native globaltol simplify."""
    from chebfunjax.chebfun1d.chebfun import chebfun

    represented = chebfun(matrix, domain=interval, pref=pref)
    panel = represented.funs[0].tech
    # A single interval: native global/local tolerance is common-max/column.
    values = panel.values
    if values.ndim == 1:
        values = values[:, None]
    scales = jnp.max(jnp.abs(values), axis=0)
    panel = panel.simplify(relative * jnp.max(scales) / scales)
    coefficients = panel.coeffs
    if coefficients.ndim == 1:
        coefficients = coefficients[:, None]
    if tech is Trigtech:
        return [
            tech.from_coeffs(coefficients[:, k], is_real=panel.real_columns[k])
            for k in range(coefficients.shape[1])
        ]
    return [tech.from_coeffs(coefficients[:, k]) for k in range(coefficients.shape[1])]


def sample_test(f, op, tolerance, vectorize):
    count = 20 if vectorize else 30
    coordinates = []
    for base, a, b in zip((2, 3, 5), f.domain[::2], f.domain[1::2]):
        sequence = []
        for index in range(1, count + 1):
            fraction, denom = 0.0, 1.0
            while index:
                denom /= base
                index, remainder = divmod(index, base)
                fraction += remainder * denom
            sequence.append(fraction)
        coordinates.append((b - a) * jnp.asarray(sequence) + a)
    if vectorize:
        coordinates = jnp.meshgrid(*coordinates, indexing="ij")
    true = evaluate(op, coordinates, vectorize)
    actual = f(*coordinates)
    error = jnp.max(jnp.abs(true - actual))
    return not bool(error > 10 * tolerance), float(error)


def construct(op, domain, pref, tech=Chebtech2, *, vectorize=False, fiber_dim=None, trace=None):
    """All three adaptive classic phases, with source stop/retry policies."""
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    from chebfunjax.chebpref import ChebfunPref

    pref = ChebfunPref(pref)
    pref.tech = tech
    # Source calls tech.techPref() without pref: these are factory defaults.
    factory = ChebfunPref.getFactoryDefaults()
    factory.tech = tech
    grid = int(factory.techPrefs.minSamples)
    max_length = int(factory.techPrefs.maxLength)
    epsilon = float(pref.cheb3Prefs.chebfun3eps)
    max_rank = int(pref.cheb3Prefs.maxRank)
    if fiber_dim is None:
        fiber_dim = dim_cluster(op, domain, vectorize, tech)
    order = (1, 2, 0) if fiber_dim == 1 else (2, 0, 1) if fiber_dim == 2 else (0, 1, 2)
    original = op

    def permuted(a, b, c):
        inputs = [None] * 3
        for axis, value in zip(order, (a, b, c)):
            inputs[axis] = value
        return original(*inputs)

    op = permuted
    domain = tuple(domain[2 * k + j] for k in order for j in (0, 1))
    is_happy, failure = False, False
    while not is_happy and not failure:
        axes = tensor_axes((grid, grid, grid), domain, tech)
        grid2d = grid
        values = _on_axes(op, axes, vectorize)
        if bool(jnp.any(jnp.isinf(values))):
            raise ValueError(
                "CHEBFUN:CHEBFUN3:constructor:inf: Function returned INF when evaluated"
            )
        if bool(jnp.any(jnp.isnan(values))):
            raise ValueError(
                "CHEBFUN:CHEBFUN3:constructor:nan: Function returned NaN when evaluated"
            )
        relative, absolute = tolerance3(values, axes, domain, epsilon, grid)
        pref.chebfuneps = relative
        skeleton = aca3(values, axes, domain, epsilon, absolute)
        while (skeleton.fail3 or skeleton.fail2) and grid < 363:
            if skeleton.fail3:
                grid = refine1(grid, tech)
                shape = (grid, grid, grid)
            else:
                grid2d = refine1(grid2d, tech)
                shape = (grid2d, grid2d, grid)
            axes = tensor_axes(shape, domain, tech)
            values = _on_axes(op, axes, vectorize)
            relative, absolute = tolerance3(values, axes, domain, epsilon, grid)
            pref.chebfuneps = relative
            skeleton = aca3(values, axes, domain, epsilon, absolute)
        phase1_metadata = dict(
            shape=list(values.shape),
            indices3=skeleton.indices3.tolist(),
            indices2=[p.tolist() for p in skeleton.indices2],
            pivots3=skeleton.pivots.tolist(),
            fail3=skeleton.fail3,
            fail2=skeleton.fail2,
        )
        if skeleton.pivots.size > max_rank:
            warnings.warn(
                "CHEBFUN:CHEBFUN3:constructor:rank: Not a low-rank function.", RuntimeWarning
            )
            failure = True
        resolved = happiness_btd(skeleton, tech, relative, pref)
        is_happy = all(resolved)
        if skeleton.pivots.size == 1 and bool(skeleton.pivots[0] == 0):
            is_happy = True
        grid_phase1 = grid
        failure = False
        phase2_count = 0
        while not is_happy and not failure:
            skeleton, axes = refine_skeleton(op, skeleton, axes, domain, tech, resolved, vectorize)
            matrices = (skeleton.blocks[0][0], skeleton.blocks[0][2], skeleton.fibers)
            resolved = [
                old or happy(
                    matrix, tech, relative, pref,
                    initial_vscale=jnp.max(jnp.abs(matrix)),
                )
                for old, matrix in zip(resolved, matrices)
            ]
            is_happy = all(resolved)
            lengths = [axis.size for axis in axes]
            if max(lengths) >= max_length:
                warnings.warn(
                    f"CHEBFUN:CHEBFUN3:constructor:notResolved: Unresolved with maximum CHEBFUN3 length: {max_length}.",
                    RuntimeWarning,
                )
                failure = True
            grid = min(axis.size for axis in axes)
            phase2_count += 1
        if (
            bool(jnp.linalg.norm(skeleton.blocks[0][0]) == 0)
            or bool(jnp.linalg.norm(skeleton.blocks[0][2]) == 0)
            or bool(jnp.linalg.norm(skeleton.fibers) == 0)
        ):
            skeleton.fibers = jnp.zeros((grid, 1))
            skeleton.blocks = [(jnp.zeros((grid, 1)), jnp.asarray([jnp.inf]), jnp.zeros((grid, 1)))]
            skeleton.pivots = jnp.asarray([jnp.inf])
            is_happy = True
        core, columns, rows = btd_to_tucker(skeleton.blocks, skeleton.pivots, absolute)
        factors = [
            _factors(matrix, interval, pref, relative, tech)
            for matrix, interval in zip(
                (columns, rows, skeleton.fibers), zip(domain[::2], domain[1::2])
            )
        ]
        result = Chebfun3(*factors, core, domain)
        sample_error = None
        if pref.cheb3Prefs.sampleTest:
            passed, sample_error = sample_test(result, op, absolute, vectorize)
            if not passed:
                is_happy = False
        if trace is not None:
            trace.append(
                dict(
                    fiber_dim=int(fiber_dim),
                    grid_phase1=int(grid_phase1),
                    phase1_shape=list(values.shape),
                    phase1=phase1_metadata,
                    relative=float(relative),
                    absolute=float(absolute),
                    pivots=skeleton.pivots.tolist(),
                    indices3=skeleton.indices3.tolist(),
                    indices2=[p.tolist() for p in skeleton.indices2],
                    phase2_iterations=phase2_count,
                    resolved=resolved,
                    lengths=[max(t.n for t in group) for group in factors],
                    rank=list(core.shape),
                    sample_error=sample_error,
                    is_happy=is_happy,
                    failure=failure,
                )
            )
        if not is_happy:
            grid = refine1(grid_phase1, tech)
    inverse = tuple(order.index(k) for k in range(3))
    return result.permute(inverse)
