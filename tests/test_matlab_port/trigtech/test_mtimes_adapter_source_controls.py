"""Supplemental source-contract controls for Trigtech ``@`` adaptation.

Native ordering and zero-row rules come from @trigtech/mtimes.m. These checks
cover adapter inputs beyond original predicates 11–14; they do not replace
those native assertions.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech


def _two_column_tech():
    return Trigtech.from_function(
        lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.cos(20 * jnp.pi * x)], axis=-1
        )
    )


def test_left_and_right_scalar_mtimes_adapters():
    f = _two_column_tech()
    for product in (f @ 2.0, f @ jnp.asarray([2.0]), 2.0 @ f, jnp.asarray([2.0]) @ f):
        np.testing.assert_array_equal(product.coeffs, f.coeffs * 2.0)
        np.testing.assert_array_equal(product.values, f.values * 2.0)


def test_complex_python_sequence_mtimes_preserves_values():
    f = _two_column_tech()
    for weights in ([1.0 + 2.0j, 3.0 - 4.0j], (1.0 - 1.5j, -2.0 + 0.25j)):
        w = jnp.asarray(weights, dtype=jnp.complex128)
        product = f @ weights
        np.testing.assert_array_equal(product.coeffs, f.coeffs @ w)
        np.testing.assert_array_equal(product.values, f.values @ w)


def test_empty_operand_precedes_mtimes_validation():
    f = _two_column_tech()
    assert (f @ Trigtech.empty()).isempty()
    assert (f @ jnp.empty((0,), dtype=jnp.uint8)).isempty()
    assert (jnp.ones((1, 3)) @ Trigtech.empty()).isempty()
    assert (Trigtech.empty() @ jnp.uint8(128)).isempty()


def test_zero_product_uses_native_single_row_storage():
    f = _two_column_tech()
    product = f @ jnp.zeros((2, 3), dtype=jnp.float64)
    assert product.coeffs.shape == (1, 3)
    assert product.values.shape == (1, 3)
    assert bool(jnp.all(product.coeffs == 0))
    assert bool(jnp.all(product.values == 0))


def test_zero_result_collapse_uses_result_scale_and_storage_isreal():
    f = Trigtech.from_function(
        lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
    )
    product = f @ jnp.zeros((2, 1), dtype=jnp.complex128)
    assert product.coeffs.shape == (1, 1)
    # MATLAB isreal(complex(zeros(...))) is false by storage type, even though
    # the values and the resulting zero coefficients have no imaginary part.
    assert not product.is_real

    # Native tests vscale of the product. Zero times Inf gives NaNs, so it
    # must not take the all-zero one-row storage branch.
    zeros = Trigtech(coeffs=jnp.zeros((7,), dtype=jnp.float64), is_real=True)
    nonfinite = zeros @ jnp.inf
    assert nonfinite.coeffs.shape[0] == 7
    assert bool(jnp.any(jnp.isnan(nonfinite.coeffs)))

