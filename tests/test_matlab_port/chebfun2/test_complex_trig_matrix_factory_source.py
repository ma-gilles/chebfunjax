"""Independent complex periodic matrix-constructor control.

The sampled matrix uses MATLAB's trig grid on a non-square physical domain.
An analytic Fourier sum supplies independent source-grid and off-grid
values through the public numeric-matrix factory path.

MATLAB source: @chebfun2/constructor.m (constructFromDouble),
               @trigtech/vals2coeffs.m, @trigtech/coeffs2vals.m
Chebfun commit: 7574c77
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import loadmat

from chebfunjax.chebfun2d.chebfun2 import chebfun2


def _complex_periodic_value(x, y, domain, nx, ny):
    xa, xb, ya, yb = domain
    tx = 2.0 * (x - xa) / (xb - xa) - 1.0
    ty = 2.0 * (y - ya) / (yb - ya) - 1.0
    value = (
        (0.7 + 1.2j)
        + (1.3 - 0.4j) * jnp.exp(1j * jnp.pi * (2.0 * tx - ty))
        + (-0.2 + 0.9j) * jnp.exp(-1j * jnp.pi * (tx + 2.0 * ty))
    )
    # For even sample counts, the Nyquist cosine is represented by the
    # single k=-N/2 coefficient and evaluated through MATLAB's even-N
    # cosine endpoint branch; a complex amplitude checks phase retention.
    if nx % 2 == 0:
        value = value + (0.35 + 0.8j) * jnp.cos((nx // 2) * jnp.pi * tx)
    if ny % 2 == 0:
        value = value + (-0.45 + 0.3j) * jnp.cos((ny // 2) * jnp.pi * ty)
    return value


@pytest.mark.parametrize(("nx", "ny"), [(11, 9), (12, 9), (11, 10), (12, 10)])
def test_public_trig_matrix_constructor_preserves_complex_fourier_modes(nx, ny):
    # Source constructFromDouble uses equispaced trigpts in each marked axis.
    # The four unequal grids cover all odd/even parity combinations.
    domain = (2.0, 7.0, -3.0, 5.0)
    xa, xb, ya, yb = domain
    xnodes = xa + (xb - xa) * jnp.arange(nx) / nx
    ynodes = ya + (yb - ya) * jnp.arange(ny) / ny
    xx, yy = jnp.meshgrid(xnodes, ynodes)
    A = _complex_periodic_value(xx, yy, domain, nx, ny)

    # This exercises chebfun2(A, domain, 'trig') numeric-data dispatch,
    # not only the direct Chebfun2.from_values helper.
    f = chebfun2(A, domain=domain, trig=True)
    assert f.approx.techs == ("trig", "trig")

    # Source samples and a non-grid tensor probe test orientation, physical
    # interval normalization, Fourier shifts, complex ACA factors and both
    # trigonometric slice transforms against the closed form.
    np.testing.assert_allclose(
        f(xx, yy),
        A,
        atol=100 * np.finfo(np.float64).eps * float(jnp.max(jnp.abs(A))),
        rtol=0,
    )
    xq = xa + (xb - xa) * jnp.asarray([0.07, 0.19, 0.37, 0.58, 0.81, 0.94])
    yq = ya + (yb - ya) * jnp.asarray([0.11, 0.29, 0.52, 0.73, 0.91])
    xgrid, ygrid = jnp.meshgrid(xq, yq)
    expected = _complex_periodic_value(xgrid, ygrid, domain, nx, ny)
    scale = float(jnp.max(jnp.abs(expected)))
    np.testing.assert_allclose(
        f(xgrid, ygrid),
        expected,
        atol=100 * np.finfo(np.float64).eps * scale,
        rtol=0,
    )


@pytest.mark.parametrize(("nx", "ny"), [(11, 9), (12, 9), (11, 10), (12, 10)])
def test_public_complex_periodic_matrix_constructor_native_capture(nx, ny):
    """Compare literal matrix dispatch and callback route to the same source field.

    Bound mapping: MATLAB test_constructor.m defines tol =
    100*pref.cheb2Prefs.chebfun2eps and uses norm(f1-f2)<tol (pass 1), plus
    2e4*tol for dense max error (pass 7). The former is used for the
    matrix-vs-callback function norm; the latter for pointwise grids. No new
    tolerance is selected for this supplemental native control.
    """
    rec = loadmat(
        (Path(__file__).resolve().parents[2] / "references" / "complex_trig_matrix_native")
        / f"case_{nx:02d}x{ny:02d}.mat",
        squeeze_me=True,
        struct_as_record=False,
    )
    dom = tuple(np.asarray(rec["domain"]).reshape(-1).tolist())
    A = np.asarray(rec["A"])
    tol = float(np.asarray(rec["source_tol"]).reshape(-1)[0])
    assert bool(np.asarray(rec["source_matrix_vs_callback_pass"]).reshape(-1)[0]), (
        "native matrix-vs-callback construction did not satisfy source pass(1) bound"
    )

    f_matrix = chebfun2(jnp.asarray(A), domain=dom, trig=True)
    f_callback = chebfun2(
        lambda x, y: _complex_periodic_value(x, y, dom, nx, ny), domain=dom, trig=True
    )
    assert A.shape == (ny, nx)
    assert np.iscomplexobj(A)
    assert f_matrix.approx.techs == ("trig", "trig")
    assert float((f_matrix - f_callback).norm()) < tol

    xnodes = np.asarray(rec["xnodes"]).reshape(-1)
    ynodes = np.asarray(rec["ynodes"]).reshape(-1)
    xx, yy = np.meshgrid(xnodes, ynodes)
    native_nodes = np.asarray(rec["native_matrix_nodes"])
    np.testing.assert_allclose(np.asarray(f_matrix(xx, yy)), native_nodes, rtol=0, atol=2e4 * tol)
    np.testing.assert_allclose(np.asarray(f_matrix(xx, yy)), A, rtol=0, atol=2e4 * tol)

    xq = np.asarray(rec["xq"]).reshape(-1)
    yq = np.asarray(rec["yq"]).reshape(-1)
    xqg, yqg = np.meshgrid(xq, yq)
    native_dense = np.asarray(rec["native_matrix_dense"])
    native_callback_dense = np.asarray(rec["native_callback_dense"])
    analytic_dense = np.asarray(rec["analytic_dense"])
    actual = np.asarray(f_matrix(xqg, yqg))
    callback_actual = np.asarray(f_callback(xqg, yqg))
    np.testing.assert_allclose(actual, native_dense, rtol=0, atol=2e4 * tol)
    np.testing.assert_allclose(callback_actual, native_callback_dense, rtol=0, atol=2e4 * tol)
    np.testing.assert_allclose(actual, analytic_dense, rtol=0, atol=2e4 * tol)
    np.testing.assert_allclose(callback_actual, analytic_dense, rtol=0, atol=2e4 * tol)
