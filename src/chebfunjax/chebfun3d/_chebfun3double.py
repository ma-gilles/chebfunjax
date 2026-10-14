"""Discrete tensor construction by source ACA and block-term compression.

Provenance
----------
MATLAB source: @chebfun3/chebfun3double.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Numerical arrays and algebra use JAX; rank decisions and list assembly are eager.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech
from chebfunjax.utils.quadrature import chebpts_ab, trigpts


def tensor_axes(shape, domain, tech, *, equi=False):
    """Selected-tech tensorGrid axes, with native ndgrid orientation.

    Provenance
    ----------
    MATLAB source: @chebtech1/tensorGrid.m, @chebtech2/tensorGrid.m,
        @trigtech/tensorGrid.m; Chebfun commit: 7574c77.
    """
    if equi:
        return [jnp.linspace(a, b, n, dtype=jnp.float64) for n, a, b in
                zip(shape, domain[::2], domain[1::2])]
    if tech is Trigtech:
        return [trigpts(n, (a, b))[0] for n, a, b in
                zip(shape, domain[::2], domain[1::2])]
    kind = 1 if tech is Chebtech1 else 2
    return [chebpts_ab(n, a, b, kind=kind) for n, a, b in
            zip(shape, domain[::2], domain[1::2])]


def tensor_tolerance(values, axes, domain, epsilon):
    """Literal getTol3D/GetTol2D finite-difference tolerance.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3double.m, getTol3D/GetTol2D.
    Chebfun commit: 7574c77.
    """
    dimensions = values.ndim
    relative = ((2 if dimensions == 3 else 1)
                * jnp.asarray(max(values.shape), dtype=jnp.float64)**(4/5) * epsilon)
    gradients = []
    for axis in range(dimensions):
        selection = tuple(slice(None) if k == axis else slice(0, n-1)
                          for k, n in enumerate(values.shape))
        difference = jnp.diff(values[selection], axis=axis)
        if difference.size:
            spacing_shape = [1] * dimensions
            spacing_shape[axis] = values.shape[axis]-1
            spacing = jnp.diff(axes[axis]).reshape(spacing_shape)
            gradients.append(jnp.max(jnp.abs(difference/spacing)))
    # Native empty gradient list is replaced by scalar 1 before domain scaling.
    gradients = jnp.stack(gradients) if gradients else jnp.asarray(1.)
    lengths = jnp.asarray(domain[1::2])-jnp.asarray(domain[::2])
    scale = jnp.maximum(jnp.max(gradients*lengths), jnp.max(jnp.abs(values)))
    return relative, scale*relative


@jax.jit
def _pivot2(matrix, tolerance):
    flat = jnp.argmax(jnp.abs(matrix).reshape(-1, order="F"))
    row, col = flat % matrix.shape[0], flat // matrix.shape[0]
    magnitude = jnp.abs(matrix[row, col])
    if matrix.shape[0] == matrix.shape[1]:
        diagonal = jnp.abs(jnp.diag(matrix))
        index = jnp.argmax(diagonal)
        use_diagonal = diagonal[index]-magnitude > -tolerance
        row = jnp.where(use_diagonal, index, row)
        col = jnp.where(use_diagonal, index, col)
        magnitude = jnp.where(use_diagonal, diagonal[index], magnitude)
    return row, col, magnitude


@jax.jit
def _subtract2(matrix, row, col):
    divided_row = jax.lax.optimization_barrier(matrix[row, :]/matrix[row, col])
    product = jax.lax.optimization_barrier(matrix[:, col, None]*divided_row[None, :])
    return matrix-product


def complete_aca2(values, tolerance):
    """Native completeACA2D with factor=0 and nonconjugating row storage.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3double.m, completeACA2D.
    Chebfun commit: 7574c77.
    """
    matrix = jnp.asarray(values)
    row, col, magnitude = _pivot2(matrix, tolerance)
    if bool(magnitude == 0):
        # Native zero input assigns scalar zero rather than shaped skeletons.
        return (jnp.zeros((1, 1)), jnp.zeros(1), jnp.zeros((1, 1)),
                jnp.zeros((1, 2), dtype=jnp.int64))
    columns, rows, pivots, locations = [], [], [], []
    for _ in range(min(matrix.shape)):
        if not bool(magnitude > tolerance):
            break
        columns.append(matrix[:, col])
        rows.append(matrix[row, :])
        pivots.append(matrix[row, col])
        locations.append(jnp.stack((row, col)))
        matrix = _subtract2(matrix, row, col)
        row, col, magnitude = _pivot2(matrix, tolerance)
    if not pivots:
        # Source initializes one zero column/row/pivot before its loop.
        return (jnp.zeros((values.shape[0], 1)),
                jnp.zeros(1),
                jnp.zeros((values.shape[1], 1)),
                jnp.zeros((1, 2), dtype=jnp.int64))
    return (jnp.stack(columns, axis=1), jnp.stack(pivots),
            jnp.stack(rows, axis=1), jnp.stack(locations))


@jax.jit
def _pivot3(tensor):
    flat = jnp.argmax(jnp.abs(tensor).reshape(-1, order="F"))
    first = flat % tensor.shape[0]
    second = flat // tensor.shape[0] % tensor.shape[1]
    third = flat // (tensor.shape[0]*tensor.shape[1])
    return first, second, third, jnp.abs(tensor[first, second, third])


@jax.jit
def _subtract3(tensor, columns, pivots, rows, fiber, pivot):
    scaled = jax.lax.optimization_barrier(columns @ jnp.diag(1/pivots))
    matrix = jax.lax.optimization_barrier(scaled @ rows.T)
    vector = jax.lax.optimization_barrier(fiber/pivot)
    product = jax.lax.optimization_barrier(matrix[:, :, None]*vector[None, None, :])
    return tensor-product


def complete_aca3(values, axes, domain, epsilon, tolerance):
    """Native completeACA3D, separating the last axis with fixed input data.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3double.m, completeACA3D.
    Chebfun commit: 7574c77.
    """
    tensor = jnp.asarray(values)
    first, second, third, magnitude = _pivot3(tensor)
    blocks, fibers, pivots, locations = [], [], [], []
    if not bool(magnitude > tolerance):
        # Native nonzero input initializes these zero arrays before its loop.
        # This differs from dropping an all-zero final 2D block after a step.
        block = (jnp.zeros((tensor.shape[0], 1)), jnp.zeros(1),
                 jnp.zeros((tensor.shape[1], 1)))
        return ([block], jnp.zeros((tensor.shape[2], 1)), jnp.zeros(1),
                jnp.zeros((1, 3), dtype=jnp.int64))
    slice_tolerance = None
    for _ in range(min(tensor.shape[2], tensor.shape[0]*tensor.shape[1])):
        if not bool(magnitude > tolerance):
            break
        fiber = tensor[first, second, :]
        matrix = tensor[:, :, third]
        pivot = tensor[first, second, third]
        if slice_tolerance is None:
            _, slice_tolerance = tensor_tolerance(matrix, axes[:2], domain[:4], epsilon)
        columns, pivots2, rows, _ = complete_aca2(matrix, slice_tolerance)
        # Source discards a final all-zero 2D pivot block after its loop.
        if bool(jnp.all(pivots2 == 0)):
            break
        blocks.append((columns, pivots2, rows))
        fibers.append(fiber)
        pivots.append(pivot)
        locations.append(jnp.stack((first, second, third)))
        tensor = _subtract3(tensor, columns, pivots2, rows, fiber, pivot)
        first, second, third, magnitude = _pivot3(tensor)
    if not blocks:
        return ([], jnp.zeros((tensor.shape[2], 0)), jnp.zeros(0),
                jnp.zeros((0, 3), dtype=jnp.int64))
    return blocks, jnp.stack(fibers, axis=1), jnp.stack(pivots), jnp.stack(locations)


def btd_to_tucker(blocks, pivots3, tolerance):
    """Native BTD-to-Tucker compression by two further complete 2D ACAs.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3double.m, btd2tucker.
    Chebfun commit: 7574c77.
    """
    if not blocks:
        return jnp.zeros((0, 0, 0)), jnp.zeros((0, 0)), jnp.zeros((0, 0))
    columns = jnp.concatenate([block[0] for block in blocks], axis=1)
    rows = jnp.concatenate([block[2] for block in blocks], axis=1)
    if columns.shape[1] > 1:
        columns_tucker, pivots, weights_columns, _ = complete_aca2(columns, tolerance)
        weights_columns = weights_columns @ jnp.diag(1/pivots)
    else:
        columns_tucker, weights_columns = columns, jnp.ones((1, 1), dtype=columns.dtype)
    if rows.shape[1] > 1:
        rows_tucker, pivots, weights_rows, _ = complete_aca2(rows, tolerance)
        weights_rows = weights_rows @ jnp.diag(1/pivots)
    else:
        rows_tucker, weights_rows = rows, jnp.ones((1, 1), dtype=rows.dtype)
    offset, slices = 0, []
    for index, (_, pivots, _) in enumerate(blocks):
        stop = offset+pivots.shape[0]
        piece = weights_columns[offset:stop, :].T @ jnp.diag(1/pivots)
        piece = piece @ weights_rows[offset:stop, :]
        slices.append(piece/pivots3[index])
        offset = stop
    core = jnp.stack(slices, axis=2)
    if bool(jnp.max(jnp.abs(columns)) == 0) or bool(jnp.max(jnp.abs(rows)) == 0):
        core = jnp.zeros((1, 1, 1))
    return core, columns_tucker, rows_tucker


def construct(values, domain, epsilon, tech=Chebtech2, *, equi=False,
              pref=None):
    """Construct numeric Chebfun3 data, preserving selected-tech fiber lengths.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3double.m; Chebfun commit: 7574c77.
    Zero tensors retain the established Python zero representation while
    native zero-path behavior is separately characterized.
    """
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3

    values = jnp.asarray(values)
    values = values.astype(jnp.result_type(values.dtype, jnp.float64))
    if values.ndim > 3:
        raise ValueError("Numeric Chebfun3 data must have at most three dimensions")
    values = values.reshape(values.shape+(1,)*(3-values.ndim))
    if values.size == 0:
        return Chebfun3.empty()
    axes = tensor_axes(values.shape, domain, tech, equi=equi)
    if bool(jnp.all(values == 0)):
        factors = [[tech.from_values(jnp.zeros(n, dtype=values.dtype))]
                   for n in values.shape]
        return Chebfun3(*factors, jnp.zeros((1, 1, 1), dtype=values.dtype), tuple(domain))
    relative_tolerance, tolerance = tensor_tolerance(
        values, axes, domain, epsilon)
    # Source completeACA3D uses the selected-tech tensorGrid for its first
    # slice tolerance, including when the original samples are equispaced.
    slice_axes = tensor_axes(values.shape[:2], domain[:4], tech) if equi else axes[:2]
    blocks, tubes, pivots, _ = complete_aca3(values, slice_axes, domain, epsilon, tolerance)
    core, columns, rows = btd_to_tucker(blocks, pivots, tolerance)
    if equi:
        # Native chebfun3double builds each factor matrix through the public
        # 1D `chebfun(values, interval, 'equi', pref)` constructor. That path
        # runs FUNQUI before the selected adaptive Tech, including Trigtech.
        from chebfunjax.chebfun1d.chebfun import chebfun
        from chebfunjax.chebpref import ChebfunPref

        factor_pref = ChebfunPref() if pref is None else ChebfunPref(pref)
        factor_pref.tech = tech
        factor_pref.enableFunqui = True
        factor_pref.chebfuneps = float(relative_tolerance)
        factors = []
        for matrix, (a, b) in zip(
                (columns, rows, tubes),
                zip(domain[::2], domain[1::2])):
            represented = chebfun(matrix, domain=(a, b), pref=factor_pref)
            group = []
            for k in range(matrix.shape[1]):
                tech_factor = represented[k].funs[0].tech
                # A scalar Chebfun's source indexing returns itself, retaining
                # its singleton value column in the Tech coefficients. Numeric
                # Chebfun3 factors store each scalar Tech as a vector.
                if matrix.shape[1] == 1:
                    coefficients = tech_factor.coeffs
                    if coefficients.ndim == 2 and coefficients.shape[1] == 1:
                        tech_factor = eqx.tree_at(
                            lambda item: item.coeffs,
                            tech_factor,
                            coefficients[:, 0],
                        )
                group.append(tech_factor)
            factors.append(group)
    else:
        factors = [[tech.from_values(matrix[:, k])
                    for k in range(matrix.shape[1])]
                   for matrix in (columns, rows, tubes)]
    return Chebfun3(*factors, core, tuple(domain))
