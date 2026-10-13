"""Source-order primitives for explicit-technology three-phase construction.

Provenance
----------
MATLAB source: @chebfun3/chebfun3f.m, ACA and phase-1 unfoldings
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax
import jax.numpy as jnp

from chebfunjax.chebfun3d._chebfun3f_tech import _grid_ratios, envelope_happiness


def unfold(tensor, axis):
    """Move the selected mode first and flatten remaining modes columnwise.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    moved = jnp.moveaxis(jnp.asarray(tensor), axis, 0)
    return moved.reshape((moved.shape[0], -1), order="F")


def decode_columns(columns, first_size):
    """Decode native ind2sub([first_size, second_size], columns).

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    columns = jnp.asarray(columns)
    return columns % first_size, columns // first_size


@jax.jit
def _pivot(matrix):
    flat = jnp.argmax(jnp.abs(matrix).reshape(-1, order="F"))
    row, column = flat % matrix.shape[0], flat // matrix.shape[0]
    return row, column, jnp.abs(matrix[row, column])


@jax.jit
def _subtract_cross(matrix, row, column):
    # Native A - A(:,J)*A(I,:)./A(I,J), including this operation order.
    product = matrix[:, column, None] * matrix[None, row, :]
    product = jax.lax.optimization_barrier(product)
    quotient = product / jnp.broadcast_to(matrix[row, column], matrix.shape)
    return matrix - quotient


def aca(matrix, tolerance, max_iterations):
    """Eager adaptive cross approximation with native first-maximum ordering.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m, ACA
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    original = jnp.asarray(matrix)
    residual = original
    rows, columns = [], []
    if residual.size:
        for _ in range(max_iterations):
            row, column, error = _pivot(residual)
            if float(error) < tolerance:
                break
            rows.append(int(row))
            columns.append(int(column))
            residual = _subtract_cross(residual, row, column)
    rows = jnp.asarray(rows, dtype=jnp.int64)
    columns = jnp.asarray(columns, dtype=jnp.int64)
    return (original[:, columns], jnp.conj(original[rows, :].T),
            original[rows[:, None], columns[None, :]], rows, columns)


def cosine_points(n, a, b):
    """Native C2 sampling grid, also used by its explicit C1 constructor.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    angle = _grid_ratios(jnp.arange(n, dtype=jnp.float64) * jnp.pi, n - 1)
    return a + (-jnp.cos(angle) + 1) * (b - a) / 2


def refine_c1_fibers(op, fibers, pairs, sizes, domain, relative_tol):
    """Literal nested phase-2 sampling with C1 envelope happiness.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m, phase2 and happinessCheck3
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.tech.chebtech import Chebtech1

    fibers = list(map(jnp.asarray, fibers))
    lengths = list(sizes)
    resolved = [envelope_happiness(values, Chebtech1, relative_tol) for values in fibers]
    lengths = [n if happy else 2*n - 1 for n, happy in zip(lengths, resolved)]
    while not all(resolved):
        grids = [cosine_points(n, a, b) for n, a, b in
                 zip(lengths, domain[::2], domain[1::2])]
        factors = [(m-1)//(n-1) for m, n in zip(lengths, sizes)]
        for axis in range(3):
            if resolved[axis]:
                continue
            remaining = [k for k in range(3) if k != axis]
            coordinates = [None, None, None]
            coordinates[axis] = grids[axis][1::2, None]
            for k, selected in zip(remaining, pairs[axis]):
                coordinates[k] = grids[k][jnp.asarray(selected) * factors[k]][None, :]
            coordinates = jnp.broadcast_arrays(*coordinates)
            added = jnp.asarray(op(*coordinates))
            old = fibers[axis]
            refined = jnp.zeros((lengths[axis], old.shape[1]), dtype=old.dtype)
            fibers[axis] = refined.at[::2].set(old).at[1::2].set(added)
            resolved[axis] = envelope_happiness(fibers[axis], Chebtech1, relative_tol)
            if not resolved[axis]:
                lengths[axis] = 2*lengths[axis] - 1
    grids = [cosine_points(n, a, b) for n, a, b in
             zip(lengths, domain[::2], domain[1::2])]
    return (*fibers, *grids)


def c1_factors(values, relative_tol, session_tol):
    """Apply the native two numeric-factor simplifications using C1 tech.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    from chebfunjax.tech.chebtech import Chebtech1

    combined = Chebtech1.from_values(jnp.asarray(values)).simplify(session_tol)
    scales = jnp.max(jnp.abs(combined.values), axis=0)
    tolerances = relative_tol * jnp.max(scales) / scales
    combined = combined.simplify(tolerances)
    return [Chebtech1.from_coeffs(combined.coeffs[:, k]) for k in range(values.shape[1])]


def deim(basis):
    """Native DEIM with square solves and first-maximum row ordering.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    indices = [int(jnp.argmax(jnp.abs(basis[:, 0])))]
    for k in range(1, basis.shape[1]):
        selected = jnp.asarray(indices)
        coefficient = jnp.linalg.solve(basis[selected, :k], basis[selected, k])
        residual = basis[:, k] - basis[:, :k] @ coefficient
        indices.append(int(jnp.argmax(jnp.abs(residual))))
    indices = jnp.asarray(indices)
    return indices, basis[indices, :]


def invtprod(tensor, first, second, third):
    """Native source-order inverse tensor products with square solves.

    Provenance
    ----------
    MATLAB source: @chebfun3/chebfun3f.m
    Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
    """
    tensor = jnp.asarray(tensor)
    for axis, matrix in enumerate((first, second, third)):
        # Source mode3 uses permutation [3,2,1]; retain its column order.
        permutation = ((0, 1, 2), (1, 0, 2), (2, 1, 0))[axis]
        moved = jnp.transpose(tensor, permutation)
        shape = moved.shape
        solved = jnp.linalg.solve(matrix, moved.reshape((shape[0], -1), order="F"))
        tensor = solved.reshape(shape, order="F").transpose(permutation)
    return tensor
