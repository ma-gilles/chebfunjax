"""Portable source controls for Spherefun addition numerical stages.

Chebfun7574c77 @spherefun/plus.m and @separableApprox/plus.m.
Expected arithmetic is literal source ordering; existing trig transform is
shared explicitly. These controls do not claim an independent FFT oracle.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun import _plus as candidate
from chebfunjax.tech import trigtech as reference


def exact(actual, expected):
    """Compare shape, dtype and all component bits, including signed zeros."""
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.tobytes() == b.tobytes(), (a, b)


_POLE_PIVOTS = [(2., -3.), (0., 2.), (-0., 2.), (np.inf, 2.),
                (-np.inf, 2.), (np.nan, 2.), (np.inf, -np.inf)]


@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('fp,gp', _POLE_PIVOTS,
                         ids=['finite', 'positive-zero', 'negative-zero',
                              'positive-inf', 'negative-inf', 'nan', 'both-inf'])
def test_literal_zero_pole_divisions_exceptional_components(disable, fp, gp):
    # Direct arithmetic diagnostic of the original zero branch. Some
    # exceptional inputs do not reach this branch through public addPoles.
    fv = jnp.asarray([0., -0., 1., -1., 2.])
    gv = jnp.asarray([-0., 0., -2., 3., -1.])
    fp, gp = jnp.asarray(fp), jnp.asarray(gp)
    with jax.disable_jit(disable):
        values = (0/fp)*fv + (0/gp)*gv
        expected = reference._trig_vals2coeffs_impl(values)
        actual = candidate._zero_pole_coefficients(fv, gv, fp, gp)
        exact(actual, expected)


@pytest.mark.parametrize('keep', [1, 2])
def test_selected_reconstruction_source_products(keep):
    # Different rectangular factors and nonsymmetric bases expose transposes,
    # wrong selected axes, and accidental use of unselected singular values.
    qc = jnp.asarray([[1., -2., 3.], [4., .5, -1.]])
    qr = jnp.asarray([[2., -1., 4.], [3., 2., -.5], [-1., 5., 2.], [4., 0., 1.]])
    u = jnp.asarray([[1., 2., -3.], [-4., 1., 2.], [2., -1., .5]])
    vh = jnp.asarray([[2., -3., 1.], [4., .5, -2.], [-1., 2., 3.]])
    s = jnp.asarray([2., 4., 8.])
    # Literal native selection precedes each matrix product and reciprocal.
    expected = qc @ u[:, :keep], qr @ vh[:keep, :].T, 1 / s[:keep]
    actual = candidate._compression_reconstruct(qc, qr, u, s, vh, keep=keep)
    assert actual[0].shape == (2, keep)
    assert actual[1].shape == (4, keep)
    assert actual[2].shape == (keep,)
    for left, right in zip(actual, expected, strict=True):
        exact(left, right)
