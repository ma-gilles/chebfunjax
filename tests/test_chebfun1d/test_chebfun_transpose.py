"""Core-suite mirror tests for row-chebfun transpose / orientation
(Fable 5).

Covers the ``isTransposed`` orientation flag and the operations that
dispatch on it: transpose/ctranspose/permute, orientation-aware size and
repr, mtimes (row*col inner product), fliplr role swap, and orientation
propagation through calculus.  These live outside ``tests/test_matlab_port``
so the feature has non-golden coverage independent of the MATLAB harness.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
XS = jnp.asarray(np.linspace(-0.9, 0.9, 41))


def _col():
    return cj.chebfun(lambda x: jnp.sin(x) * (x - 0.1))


def _arr():
    return cj.chebfun(
        lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1),
        domain=(-1, -0.5, 0, 0.5, 1),
    )


class TestOrientationFlag:
    def test_default_is_column(self):
        assert not _col().is_transposed

    def test_transpose_flips(self):
        f = _col()
        assert f.T.is_transposed
        assert not f.T.T.is_transposed

    def test_transpose_shares_values(self):
        f = _col()
        # Scalar row eval is identical to the column eval (1-D transpose).
        assert float(jnp.max(jnp.abs(f.T(XS) - f(XS)))) == 0.0

    def test_ctranspose_conjugates(self):
        f = cj.chebfun(lambda x: jnp.exp(1j * x))
        h = f.H
        assert h.is_transposed
        assert float(jnp.max(jnp.abs(h(XS) - jnp.conj(f(XS))))) < 1e3 * EPS

    def test_marker_does_not_leak_to_column(self):
        # Building a fresh column after a transpose must not inherit the flag.
        f = _col()
        _ = f.T
        assert not cj.chebfun(jnp.sin).is_transposed


class TestPermute:
    def test_identity(self):
        f = _col()
        assert not f.permute([1, 2]).is_transposed

    def test_transpose(self):
        f = _col()
        assert f.permute([2, 1]).isequal(f.T)

    @pytest.mark.parametrize("order", [[1, 1], [2, 2], [1, 3]])
    def test_invalid(self, order):
        with pytest.raises(ValueError):
            _col().permute(order)


class TestSizeAndRepr:
    def test_column_size(self):
        assert _col().size() == (float("inf"), 1)
        assert _arr().size() == (float("inf"), 3)

    def test_row_size(self):
        assert _col().T.size() == (1, float("inf"))
        assert _arr().T.size() == (3, float("inf"))
        assert _arr().T.size(1) == 3
        assert _arr().T.size(2) == float("inf")

    def test_repr_orientation(self):
        assert repr(_col()).startswith("   chebfun column")
        assert repr(_col().T).startswith("   chebfun row")


class TestMtimesDispatch:
    def test_row_times_column_is_inner_product(self):
        f = cj.chebfun(jnp.sin, domain=(0.0, float(np.pi)))
        g = cj.chebfun(lambda x: x, domain=(0.0, float(np.pi)))
        ip = f.T * g
        # int_0^pi x sin x dx = pi
        assert abs(float(ip) - np.pi) < 1e3 * EPS
        assert abs(float(ip) - float(f.innerProduct(g))) < 1e3 * EPS

    def test_ctranspose_times_column_conjugates(self):
        f = cj.chebfun(lambda x: jnp.exp(1j * x), domain=(0.0, float(np.pi)))
        g = cj.chebfun(lambda x: x + 0.0j, domain=(0.0, float(np.pi)))
        # f' * g == <f, g> == int conj(f) g.
        val = f.H * g
        assert abs(complex(val) - complex(f.innerProduct(g))) < 1e3 * EPS

    def test_same_orientation_is_pointwise(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        prod = f.T * g.T
        assert prod.is_transposed
        assert float(jnp.max(jnp.abs(prod(XS) - (f(XS) * g(XS))))) < 1e3 * EPS

    def test_column_times_row_unsupported(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        with pytest.raises(NotImplementedError):
            _ = f * g.T

    def test_scalar_scaling_preserves_orientation(self):
        f = _col().T
        assert (3.0 * f).is_transposed
        assert (f * 2.0).is_transposed


class TestFliplr:
    def test_row_reflects_about_midpoint(self):
        f = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(3 * x))
        g = f.T.fliplr()
        assert g.is_transposed
        assert float(jnp.max(jnp.abs(g(XS) - f(-XS)))) < 1e3 * EPS * f.vscale

    def test_row_fliplr_involution(self):
        f = _col()
        assert f.T.fliplr().fliplr().isequal(f.T)

    def test_column_array_reverses(self):
        # Column array-valued fliplr still reverses column order (unchanged).
        g = _arr()
        gg = g.fliplr()
        assert not gg.is_transposed
        assert gg.fliplr().isequal(g)


class TestOrientationPropagation:
    def test_diff_keeps_orientation(self):
        assert _col().T.diff().is_transposed

    def test_cumsum_keeps_orientation(self):
        assert _col().T.cumsum().is_transposed

    def test_negation_keeps_orientation(self):
        assert (-_col().T).is_transposed

    def test_conj_keeps_orientation(self):
        assert _col().T.conj().is_transposed

    def test_isequal_orientation_sensitive(self):
        f = _col()
        assert not f.isequal(f.T)
        assert f.T.isequal(f.T)
