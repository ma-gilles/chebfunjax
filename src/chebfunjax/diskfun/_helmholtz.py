"""JAX coefficient solve for the disk ultraspherical Dirichlet problem.

Provenance
----------
MATLAB source : @diskfun/helmholtz.m
Chebfun commit: 7574c77
"""
from functools import partial

import jax
import jax.numpy as jnp

from chebfunjax.spherefun._banded_lu import solve_banded


@jax.jit
def _border_correction(inv_b, inv_w):
    """Source Sherman-Morrison correction for independent Fourier systems."""
    # Separate source multiply/divide/subtract operations into scan iterations.
    # A product barrier alone still permits reciprocal-based division, which
    # can flush 1/denominator and change a normal quotient into zero.
    numerator = jnp.broadcast_to(jnp.sum(inv_b[:, 1:], axis=1)[:, None], inv_w.shape)
    denominator = jnp.broadcast_to((1 + jnp.sum(inv_w[:, 1:], axis=1))[:, None], inv_w.shape)
    operands = jnp.stack((inv_w, denominator, inv_b))

    def step(value, instruction):
        opcode, operand = instruction
        result = jax.lax.switch(opcode, (
            lambda: value * operand,
            lambda: value / operand,
            lambda: operand - value,
        ))
        return result, None

    return jax.lax.scan(step, numerator, (jnp.arange(3), operands), unroll=False)[0]


@partial(jax.jit, static_argnames=('parity',))
def _solve_parity(operator, conversion, forcing, boundary, modes, indices, *, parity):
    """Solve one parity's independent Fourier systems and source rank-one border."""
    radial = operator[parity::2, parity::2]
    convert = conversion[parity::2, parity::2]
    interior = radial[None, :, :] - modes[:, None, None]**2 * convert[None, :, :]
    count, _, width = interior.shape
    first = jnp.zeros((count, 1, width), dtype=interior.dtype).at[:, 0, 0].set(1)
    systems = jnp.concatenate((first, interior), axis=1)
    rhs = jnp.concatenate((boundary[indices, None], forcing[indices, parity:-2:2]), axis=1)
    weight = jnp.zeros_like(rhs).at[:, 0].set(1)
    # Exact source support is lower=upper=2 after prepending the first row.
    # Keep dense source assembly, but factor only its five nonzero diagonals.
    # The last two storage columns reserve upper fill from distance-two pivots.
    rows = jnp.arange(width)[:, None]
    columns = rows + jnp.arange(-2, 5)[None, :]
    valid = (columns >= 0) & (columns < width) & (columns <= rows + 2)
    bands = jnp.where(valid[None], systems[:, rows, jnp.clip(columns, 0, width-1)], 0)
    inverse, _ = jax.vmap(lambda a, b: solve_banded(a, b, lower=2, upper=2))(
        bands, jnp.stack((rhs, weight), axis=-1))
    inv_b, inv_w = inverse[..., 0], inverse[..., 1]
    return _border_correction(inv_b, inv_w)


@partial(jax.jit, static_argnames=('real_valued',))
def solve_modes(operator, conversion, forcing, boundary, real_valued):
    """Source Fourier decoupling, parity reduction and conjugate completion.

    Provenance
    ----------
    MATLAB source : @diskfun/helmholtz.m
    Chebfun commit: 7574c77
    """
    n, m = forcing.shape
    count = n//2 + 1 if real_valued else n
    indices = jnp.arange(count)
    modes = -n//2 + indices
    result = jnp.zeros((m, n), dtype=jnp.result_type(operator, forcing, boundary, 1j))
    for parity in (0, 1):
        # Fourier modes are independent; group equal radial system sizes.
        selected = indices[(parity + n//2) % 2::2]
        solved = _solve_parity(operator, conversion, forcing, boundary,
                               modes[selected], selected, parity=parity)
        result = result.at[parity::2, selected].set(solved.T)
    if real_valued:
        result = result.at[:, count:n].set(jnp.conj(result[:, 1:count-1][:, ::-1]))
    return result


def assemble_operator(m, K):
    """Source ultraspherical operator; retain every complex component of K².

    Provenance
    ----------
    MATLAB source : @diskfun/helmholtz.m
    Chebfun commit: 7574c77
    """
    # Normalize before JIT argument abstractification: Python integers outside
    # int64 can still represent finite source-double wavenumbers.
    dtype = jnp.complex128 if jnp.iscomplexobj(K) else jnp.float64
    return _assemble_operator(m, jnp.asarray(K, dtype=dtype))


@partial(jax.jit, static_argnames=('m',))
def _assemble_operator(m, K):
    """Build source operators after normalizing the wavenumber at entry."""
    from chebfunjax.discretization import ultras

    # These source operators depend only on static resolution. The shared
    # multmat dispatcher inspects its coefficient values in Python.
    with jax.ensure_compile_time_eval():
        first = ultras.diffmat(m, 1)
        second = ultras.diffmat(m, 2)
        radial = ultras.multmat(m, jnp.asarray([0., 1.]), 1)
        radial_squared = ultras.multmat(m, jnp.asarray([.5, 0., .5]), 2)
        cheb_radial_squared = ultras.multmat(m, jnp.asarray([.5, 0., .5]), 0)
        conversion = ultras.convertmat(m, 0, 1)
        adjacent_conversion = ultras.convertmat(m, 1, 1)
        differential = radial_squared @ second + adjacent_conversion @ radial @ first
    operator = differential + (K**2 * conversion) @ cheb_radial_squared
    return operator[:-2], conversion, cheb_radial_squared
