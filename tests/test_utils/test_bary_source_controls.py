"""API and JAX controls for the restored bary input shapes.
Provenance
----------
MATLAB source: bary.m and tests/misc/test_bary.m.
Chebfun commit: 7574c77.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.interpolation import bary, cheb_bary_weights
from chebfunjax.utils.quadrature import chebpts


def run_controls():
    xk = chebpts(16)
    v = cheb_bary_weights(16)
    f = xk**4
    rows = []
    for shape in [(), (7,), (1, 7), (7, 1), (2, 3), (2, 3, 4)]:
        x = jnp.linspace(-0.8, 0.8, int(np.prod(shape)) if shape else 1).reshape(shape)
        got = jax.jit(lambda xx: bary(xx, f))(x)
        assert got.shape == shape
        assert float(jnp.max(jnp.abs(got - x**4))) < 1e-14
        rows.append({"control": "jit_default_nodes_" + str(shape), "status": "pass"})
    col = jnp.stack((xk**2, xk**3), axis=1)
    x = jnp.array([[-0.6, 0.2, 0.7], [-0.3, 0.4, 0.9]])
    expected = jnp.concatenate((x**2, x**3), axis=1)
    got = jax.jit(bary)(x, col, xk, v)
    assert got.shape == (2, 6)
    assert float(jnp.max(jnp.abs(got - expected))) < 1e-14
    rows.append({"control": "jit_horizontal_blocks", "status": "pass"})
    got = bary(xk, col, xk, v)
    assert bool(jnp.all(got == col))
    rows.append({"control": "exact_nodes_multiple_columns", "status": "pass"})
    col1 = (xk**4).reshape(-1, 1)
    assert bary(x, col1, xk, v).shape == x.shape
    rows.append({"control": "one_column_scalar_shape", "status": "pass"})
    try:
        bary(jnp.zeros((2, 3, 4)), col, xk, v)
    except ValueError as e:
        assert str(e) == "CHEBFUN:bary:evalArrayAtNDArray"
    else:
        raise AssertionError("missing native array-polynomial ND rejection")
    rows.append({"control": "native_nd_array_rejection", "status": "pass"})
    deriv = jax.grad(lambda t: bary(t, f, xk, v))(jnp.array(0.37))
    assert abs(float(deriv) - 4 * 0.37**3) < 1e-12
    rows.append({"control": "scalar_autodiff_nonnodal", "status": "pass"})
    return rows


def test_bary_source_api_controls():
    run_controls()


@pytest.mark.parametrize("count", [31, 32])
def test_bary_complex_jvp_across_source_loop_boundary(count):
    # n=8 selects point/node loops immediately below/at4*n queries.
    # Analytic cubic values and directional derivatives test both loops.
    nodes = chebpts(8)
    values = nodes**3-2*nodes+1
    weights = cheb_bary_weights(8)
    x = jnp.linspace(-.8, .8, count)+.1j
    direction = jnp.full_like(x, .5+.25j)
    evaluate = jax.jit(lambda z: bary(z, values, nodes, weights))
    actual, tangent = jax.jvp(evaluate, (x,), (direction,))
    np.testing.assert_allclose(actual, x**3-2*x+1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tangent, (3*x*x-2)*direction, rtol=1e-12, atol=1e-12)
