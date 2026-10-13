"""Python-only Tucker bridges retained by the dense Chebfun3T migration."""

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3t import Chebfun3T
from chebfunjax.tech.chebtech import Chebtech2

DOMAIN = (-2.0, 4.0, 1.0, 5.0, -3.0, 1.0)


def _factor(coeffs):
    return Chebtech2.from_coeffs(jnp.asarray(coeffs, dtype=jnp.float64))


def _known_tucker():
    # Coefficients [a0, a1] encode a0 + a1*T1(t) on each reference axis.
    cols = [_factor([1.0, 1.0]), _factor([2.0, -1.0])]
    rows = [_factor([1.0, 2.0]), _factor([3.0, 1.0])]
    tubes = [_factor([2.0, -1.0]), _factor([-1.0, 2.0])]
    core = jnp.zeros((2, 2, 2), dtype=jnp.float64)
    core = core.at[0, 0, 0].set(1.25).at[1, 1, 1].set(-0.5)
    return core, cols, rows, tubes


def _analytic(x, y, z):
    t = (2 * x - (DOMAIN[0] + DOMAIN[1])) / (DOMAIN[1] - DOMAIN[0])
    u = (2 * y - (DOMAIN[2] + DOMAIN[3])) / (DOMAIN[3] - DOMAIN[2])
    v = (2 * z - (DOMAIN[4] + DOMAIN[5])) / (DOMAIN[5] - DOMAIN[4])
    x0, x1 = 1 + t, 2 - t
    y0, y1 = 1 + 2 * u, 3 + u
    z0, z1 = 2 - v, -1 + 2 * v
    return 1.25 * x0 * y0 * z0 - 0.5 * x1 * y1 * z1


def _analytic_grad(x, y, z):
    t = (2 * x - (DOMAIN[0] + DOMAIN[1])) / (DOMAIN[1] - DOMAIN[0])
    u = (2 * y - (DOMAIN[2] + DOMAIN[3])) / (DOMAIN[3] - DOMAIN[2])
    v = (2 * z - (DOMAIN[4] + DOMAIN[5])) / (DOMAIN[5] - DOMAIN[4])
    x0, x1 = 1 + t, 2 - t
    y0, y1 = 1 + 2 * u, 3 + u
    z0, z1 = 2 - v, -1 + 2 * v
    return jnp.asarray([
        (1.25 * y0 * z0 + 0.5 * y1 * z1) / 3.0,
        1.25 * x0 * z0 - 0.25 * x1 * z1,
        -0.625 * x0 * y0 - 0.5 * x1 * y1,
    ])


def _check_values_and_gradient(f):
    x, y, z = 0.37, 2.2, -1.4
    got = f(x, y, z)
    np.testing.assert_allclose(got, _analytic(x, y, z), rtol=2e-13, atol=2e-13)
    got_grad = jnp.stack([
        jax.grad(lambda q: f(q, y, z))(x),
        jax.grad(lambda q: f(x, q, z))(y),
        jax.grad(lambda q: f(x, y, q))(z),
    ])
    np.testing.assert_allclose(
        got_grad, _analytic_grad(x, y, z), rtol=3e-12, atol=3e-12
    )


def test_tucker_bridge_and_dense_conversion_preserve_data_and_inspection_views():
    core, cols, rows, tubes = _known_tucker()
    native_py = Chebfun3(cols, rows, tubes, core, DOMAIN)

    from_arrays = Chebfun3T.from_arrays(core, cols, rows, tubes, DOMAIN)
    from_chebfun = Chebfun3T.from_chebfun3(native_py)
    assert from_arrays.domain == DOMAIN == from_chebfun.domain
    np.testing.assert_array_equal(from_arrays.core, core)
    np.testing.assert_array_equal(from_chebfun.core, core)
    np.testing.assert_array_equal(from_arrays.core_numpy(), core)
    assert from_arrays.rank == (2, 2, 2)
    assert len(from_arrays.cols) == len(from_arrays.rows) == len(from_arrays.tubes) == 2
    for got, expected in zip(from_arrays.col_coeffs(), [f.coeffs for f in cols]):
        np.testing.assert_array_equal(got, expected)
    for got, expected in zip(from_arrays.row_coeffs(), [f.coeffs for f in rows]):
        np.testing.assert_array_equal(got, expected)
    for got, expected in zip(from_arrays.tube_coeffs(), [f.coeffs for f in tubes]):
        np.testing.assert_array_equal(got, expected)

    roundtrip = from_arrays.to_chebfun3()
    np.testing.assert_array_equal(roundtrip.core, core)
    np.testing.assert_allclose(roundtrip.chebcoeffs3(), from_arrays.coeffs,
                               rtol=3e-14, atol=3e-14)
    _check_values_and_gradient(from_arrays)
    _check_values_and_gradient(from_chebfun)

    # Dense-origin objects expose an exact identity-factor Tucker view. This
    # checks the extension without invoking an SVD or truncating coefficients.
    dense = Chebfun3T.from_coeffs(from_arrays.coeffs, DOMAIN, from_arrays.vscale)
    dense_view = dense.to_chebfun3()
    assert dense.rank == dense.coeffs.shape
    np.testing.assert_allclose(dense_view.chebcoeffs3(), dense.coeffs,
                               rtol=3e-14, atol=3e-14)
    _check_values_and_gradient(dense)
