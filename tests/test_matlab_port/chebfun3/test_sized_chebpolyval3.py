"""Focused source-adapter controls for sized chebpolyval3."""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.utils.quadrature import chebpts


def _polynomial():
    coeffs = jnp.zeros((3, 2, 1), dtype=jnp.float64)
    coeffs = coeffs.at[0, 0, 0].set(1.0)
    coeffs = coeffs.at[1, 0, 0].set(0.25)
    coeffs = coeffs.at[0, 1, 0].set(-0.5)
    coeffs = coeffs.at[2, 0, 0].set(0.125)
    return Chebfun3.from_coeffs(coeffs)


def test_sized_chebpolyval3_pads_each_tucker_mode():
    f = _polynomial()
    values = f.chebpolyval3(5, 6, 7)
    assert values.shape == (5, 6, 7)
    x, y = chebpts(5), chebpts(6)
    exact = (1.0 + .25*x[:, None, None]
             + .125*(2*x[:, None, None]**2-1)
             - .5*y[None, :, None]) * jnp.ones((1, 1, 7))
    # Supplementary polynomial oracle, original coefficient norm bound.
    assert float(jnp.linalg.norm((values-exact).ravel())) < 1000*jnp.finfo(jnp.float64).eps
    truncated = f.chebpolyval3(1, 6, 7)
    expected = (1.0-.5*y[None, :, None])*jnp.ones((1, 1, 7))
    assert float(jnp.linalg.norm((truncated-expected).ravel())) < 1000*jnp.finfo(jnp.float64).eps


def test_chebpolyval3_noarg_uses_representation_lengths():
    f = _polynomial()
    m, n, p = f.length()
    values = f.chebpolyval3()
    assert values.shape == (m, n, p)
    assert jnp.array_equal(values, f.chebpolyval3(m, n, p))


def test_chebpolyval3_uses_current_class_tech_and_complex_coefficients(monkeypatch):
    f = Chebfun3.from_coeffs(jnp.asarray([[[1.+2.j]], [[.25-.5j]]]))
    monkeypatch.setattr(ChebfunPref, "_defaults", ChebfunPref({"tech": Chebtech1}))
    values = f.chebpolyval3(5, 6, 7)
    assert values.shape == (5, 6, 7)
    x = chebpts(5, kind=1)
    expected = (1.+2.j + (.25-.5j)*x[:, None, None])*jnp.ones((1, 6, 7))
    assert float(jnp.linalg.norm((values-expected).ravel())) < 1000*jnp.finfo(jnp.float64).eps


@pytest.mark.parametrize("dims", [(5, None, 7), (5.5, 6, 7), (True, 6, 7)])
def test_sized_chebpolyval3_rejects_invalid_dimensions(dims):
    f = _polynomial()
    with pytest.raises((TypeError, ValueError)):
        f.chebpolyval3(*dims)


def test_native_zero_sized_grid_and_empty_object():
    # R2025b retained edge_checkpoint.mat: a nonempty f with sizes (0,2,3)
    # returns an empty tensor of that shape. Empty-object output is adapted
    # to the package's established empty-vector representation.
    f = _polynomial()
    result = f.chebpolyval3(0, 2, 3)
    assert result.shape == (0, 2, 3)
    assert result.size == 0
    assert Chebfun3.empty().chebpolyval3(0, 0, 0).size == 0


def test_native_four_output_tensor_factors():
    f = _polynomial()
    core, a, b, c = f.chebpolyval3(5, 6, 7, low_rank=True)
    assert (a.shape[0], b.shape[0], c.shape[0]) == (5, 6, 7)
    assert (a.shape[1], b.shape[1], c.shape[1]) == core.shape
    reconstructed = jnp.einsum("abc,ia,jb,kc->ijk", core, a, b, c)
    x, y = chebpts(5), chebpts(6)
    expected = (1.+.25*x[:, None, None]+.125*(2*x[:, None, None]**2-1)
                -.5*y[None, :, None])*jnp.ones((1, 1, 7))
    assert float(jnp.linalg.norm((reconstructed-expected).ravel())) < 1000*jnp.finfo(jnp.float64).eps
