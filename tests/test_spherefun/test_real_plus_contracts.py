"""Independent factor arithmetic controls; no expected constructor coefficients.

Provenance
----------
MATLAB source : @spherefun/plus.m, @trigtech/qr.m,
    @separableApprox/uminus.m
Chebfun commit: 7574c77
The small constant control is not evidence the old public subtraction snaps it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._plus import real_trig_matrix_qr
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech, _trig_vals2coeffs_impl


def constant(value):
    return Spherefun(
        cols=[Trigtech(coeffs=jnp.asarray([value]), is_real=True)],
        rows=[Trigtech(coeffs=jnp.ones(1), is_real=True)],
        pivots=jnp.ones(1),
        idx_plus=(0,),
        idx_minus=(),
        nonzero_poles=True,
        pivot_locations=((0.0, 0.0),),
    )


def test_small_exact_constant_survives():
    h = constant(1 + 2.0**-32) - constant(1.0)
    np.testing.assert_array_equal(h(jnp.asarray([-0.7, 0.2]), jnp.asarray([0.4, 2.0])), 2.0**-32)


def test_empty_identity_and_structural_negation():
    f = constant(3.0)
    assert Spherefun.empty() + f is f
    assert f + Spherefun.empty() is f
    g = -f
    assert g.cols[0] is f.cols[0]
    assert g.rows[0] is f.rows[0]
    np.testing.assert_array_equal(g.pivots, -f.pivots)
    assert g.pivot_locations == f.pivot_locations
    assert g.nonzero_poles == f.nonzero_poles


@pytest.mark.parametrize("disabled", [False, True])
def test_continuous_qr_reconstruction_and_gram(disabled):
    # Real linearly independent functions 1,cos(theta),sin(theta), over[-pi,pi].
    theta = jnp.pi * (-1 + 2 * jnp.arange(9) / 9)
    values = jnp.stack([jnp.ones_like(theta), jnp.cos(theta), jnp.sin(theta)], axis=1)
    c = _trig_vals2coeffs_impl(values)
    with jax.disable_jit(disabled):
        qc, r = real_trig_matrix_qr(c)
    np.testing.assert_allclose(qc @ r, c, rtol=0, atol=30 * np.finfo(float).eps)
    gram = 2 * jnp.pi * jnp.conj(qc).T @ qc
    np.testing.assert_allclose(gram, jnp.eye(3), rtol=0, atol=30 * np.finfo(float).eps)


def test_pole_cancellation_and_argument_order():
    # Construct exact cos(theta) and cos(theta)^3 Fourier factors independently.
    row = Trigtech(coeffs=jnp.ones(1), is_real=True)

    def pole(coeffs):
        return Spherefun(
            cols=[Trigtech(coeffs=jnp.asarray(coeffs), is_real=True)],
            rows=[row],
            pivots=jnp.ones(1),
            idx_plus=(0,),
            idx_minus=(),
            nonzero_poles=True,
            pivot_locations=((0.0, 0.0),),
        )

    f = pole([0.5, 0.0, 0.5])
    g = pole([0.125, 0.0, 0.375, 0.0, 0.375, 0.0, 0.125])
    h = f - g
    assert not h.nonzero_poles
    assert (h + f).nonzero_poles
    assert (f + h).nonzero_poles
    theta = jnp.asarray([0.3, 0.9, 2.2])
    expected = jnp.cos(theta) - jnp.cos(theta) ** 3
    np.testing.assert_allclose(
        h(jnp.zeros(3), theta), expected, rtol=0, atol=40 * np.finfo(float).eps
    )


@pytest.mark.parametrize("size", [9, 10])
def test_scale_sampling_aliases_high_modes(size):
    from chebfunjax.spherefun._plus import _sample

    # Analytic cos(13*pi*x), with stored33coeffs exceeding either sample grid.
    c = jnp.zeros(33, dtype=jnp.complex128).at[3].set(0.5).at[29].set(0.5)
    t = Trigtech(coeffs=c, is_real=True)
    actual = _sample([t], size)[:, 0]
    x = -1 + 2 * jnp.arange(size) / size
    np.testing.assert_allclose(
        actual, jnp.cos(13 * jnp.pi * x), rtol=0, atol=100 * np.finfo(float).eps
    )


def test_complex_objects_are_explicitly_outside_real_adapter():
    from chebfunjax.spherefun._plus import eligible

    f = Spherefun(
        cols=[Trigtech(coeffs=jnp.asarray([1 + 2j]), is_real=False)],
        rows=[Trigtech(coeffs=jnp.ones(1), is_real=True)],
        pivots=jnp.ones(1),
        idx_plus=(0,),
        idx_minus=(),
        nonzero_poles=True,
    )
    assert not eligible(f)
