"""Source controls for native Chebfun3 sum3 scaling and mode order.

Provenance
----------
MATLAB source: @chebfun3/sum3.m, @chebfun3/txm.m,
    @chebfun/sum.m, @bndfun/sum.m, @chebtech/sum.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech2

TOL = 1e4 * float(jnp.finfo(jnp.float64).eps)


def _factors_with_integrals(values):
    """Constant complex Chebtech2 factors with prescribed reference sums."""
    return [
        Chebtech2.from_coeffs(jnp.asarray([value / 2], dtype=jnp.complex128))
        for value in values
    ]


def _native_txm(tensor, factor, mode):
    """Literal NumPy unfold/matmul/fold for native MATLAB txm in 3-D."""
    tensor = np.asarray(tensor)
    factor = np.asarray(factor).reshape(-1)
    nx, ny, nz = tensor.shape
    if mode == 1:
        unfolded = tensor.reshape((nx, ny * nz), order="F")
        product = factor.reshape((1, nx)) @ unfolded
        return product.reshape((1, ny, nz), order="F")
    if mode == 2:
        unfolded = tensor.transpose((0, 2, 1)).reshape((nx * nz, ny), order="F")
        product = unfolded @ factor.reshape((ny, 1))
        folded = product.reshape((nx, nz, 1), order="F")
        return folded.transpose((0, 2, 1))
    if mode == 3:
        unfolded = tensor.reshape((nx * ny, nz), order="F")
        product = unfolded @ factor.reshape((nz, 1))
        return product.reshape((nx, ny, 1), order="F")
    raise ValueError(f"invalid mode {mode}")


def _native_sum3(core, ix, iy, iz):
    """Independent native sum3 composition using sequential NumPy txm."""
    result = _native_txm(core, ix, 1)
    result = _native_txm(result, iy, 2)
    result = _native_txm(result, iz, 3)
    return result.reshape(())


def _native_sum3_zyx(core, ix, iy, iz):
    """Reverse-order native unfold/matmul/fold control."""
    result = _native_txm(core, iz, 3)
    result = _native_txm(result, iy, 2)
    result = _native_txm(result, ix, 1)
    return result.reshape(())


def test_unequal_rank_complex_sum3_uses_physical_factor_integrals():
    # Unequal mode ranks ensure each sequential contraction uses its intended
    # axis. The nonunit widths exercise native per-factor physical scaling.
    domain = (-2.0, 4.0, -3.0, 1.0, 0.0, 6.0)
    scales = (3.0, 2.0, 3.0)
    core = (jnp.arange(24, dtype=jnp.float64).reshape((2, 3, 4))
            + 1j * jnp.arange(24, 0, -1, dtype=jnp.float64).reshape((2, 3, 4)))
    x_ref = jnp.asarray([0.5 + 0.25j, -1.0 + 0.5j])
    y_ref = jnp.asarray([1.0 - 0.5j, 0.25 + 0.75j, -0.5 + 0.25j])
    z_ref = jnp.asarray([0.75 + 0.5j, -0.25 + 0.5j,
                         1.0 + 0.25j, 0.5 - 0.75j])
    f = Chebfun3(
        cols=_factors_with_integrals(x_ref),
        rows=_factors_with_integrals(y_ref),
        tubes=_factors_with_integrals(z_ref), core=core, domain=domain)
    expected = _native_sum3(np.asarray(core), np.asarray(x_ref) * scales[0],
                            np.asarray(y_ref) * scales[1],
                            np.asarray(z_ref) * scales[2])
    assert abs(f.sum3() - expected) < TOL


def test_unequal_rank_complex_cancellation_exposes_sum3_order_and_scaling():
    # For x->y->z, x-mode cancellation is accumulated separately at each z
    # index. Reversing to z->y->x, or applying the x scale after contraction,
    # retains the small complex terms and gives a different result.
    core = jnp.zeros((2, 3, 4), dtype=jnp.complex128)
    core = core.at[0, 0, 0].set(1e16 + 1e16j)
    core = core.at[1, 0, 0].set(1.0 + 1.0j)
    core = core.at[0, 0, 1].set(-1e16 - 1e16j)
    x_ref = jnp.asarray([1.0 + 0.0j, 1.0 + 0.0j])
    y_ref = jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    z_ref = jnp.asarray([1.0 + 0.0j, 1.0 + 0.0j,
                         0.0 + 0.0j, 0.0 + 0.0j])
    domain = (0.0, 3.0, -1.0, 1.0, -1.0, 1.0)
    sx, sy, sz = 1.5, 1.0, 1.0
    f = Chebfun3(
        cols=_factors_with_integrals(x_ref),
        rows=_factors_with_integrals(y_ref),
        tubes=_factors_with_integrals(z_ref), core=core, domain=domain)

    native_order = _native_sum3(np.asarray(core), np.asarray(x_ref) * sx,
                                np.asarray(y_ref) * sy, np.asarray(z_ref) * sz)
    reordered = _native_sum3_zyx(np.asarray(core), np.asarray(x_ref) * sx,
                                 np.asarray(y_ref) * sy, np.asarray(z_ref) * sz)
    postscaled = _native_sum3(np.asarray(core), np.asarray(x_ref),
                              np.asarray(y_ref), np.asarray(z_ref)) * sx * sy * sz
    assert abs(native_order - (2.0 + 2.0j)) < TOL
    assert abs(native_order - reordered) > 0.5
    assert abs(native_order - postscaled) > 1.0
    assert abs(f.sum3() - native_order) < TOL
