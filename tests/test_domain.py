"""Tests for chebfunjax.domain — Domain class.

JAX contract: forward_map/inverse_map are jit-safe (domain is static).
No MATLAB refs needed (pure Python logic).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.domain import Domain

# ======================================================================
# Construction
# ======================================================================


class TestConstruction:
    """Tests for Domain construction and validation."""

    def test_basic_interval(self):
        d = Domain((-1.0, 1.0))
        assert d.breakpoints == (-1.0, 1.0)
        assert d.a == -1.0
        assert d.b == 1.0

    def test_from_endpoints(self):
        d = Domain.from_endpoints(0.0, 5.0)
        assert d.breakpoints == (0.0, 5.0)
        assert d.a == 0.0
        assert d.b == 5.0

    def test_piecewise(self):
        d = Domain((-1.0, 0.0, 1.0))
        assert d.breakpoints == (-1.0, 0.0, 1.0)
        assert d.a == -1.0
        assert d.b == 1.0

    def test_many_breakpoints(self):
        bp = (-2.0, -1.0, 0.0, 1.0, 2.0)
        d = Domain(bp)
        assert d.breakpoints == bp

    def test_from_list(self):
        d = Domain([-1, 1])
        assert d.breakpoints == (-1.0, 1.0)
        assert isinstance(d.breakpoints[0], float)

    def test_from_ints(self):
        d = Domain((0, 10))
        assert d.breakpoints == (0.0, 10.0)

    def test_error_single_point(self):
        with pytest.raises(ValueError, match="at least 2 breakpoints"):
            Domain((1.0,))

    def test_error_empty(self):
        with pytest.raises(ValueError, match="at least 2 breakpoints"):
            Domain(())

    def test_error_not_increasing(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Domain((1.0, 0.0))

    def test_error_equal_points(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Domain((0.0, 0.0))

    def test_error_not_increasing_piecewise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Domain((-1.0, 0.5, 0.3, 1.0))


# ======================================================================
# Properties
# ======================================================================


class TestProperties:
    """Tests for Domain properties."""

    def test_n_intervals_simple(self):
        assert Domain((-1.0, 1.0)).n_intervals == 1

    def test_n_intervals_piecewise(self):
        assert Domain((-1.0, 0.0, 1.0)).n_intervals == 2

    def test_n_intervals_multi(self):
        assert Domain((-2.0, -1.0, 0.0, 1.0, 2.0)).n_intervals == 4

    def test_support(self):
        d = Domain((-1.0, 0.0, 1.0))
        assert d.support == (-1.0, 1.0)

    def test_intervals_single(self):
        d = Domain((-1.0, 1.0))
        subs = list(d.intervals)
        assert len(subs) == 1
        assert subs[0] == Domain((-1.0, 1.0))

    def test_intervals_piecewise(self):
        d = Domain((-1.0, 0.0, 1.0))
        subs = list(d.intervals)
        assert len(subs) == 2
        assert subs[0] == Domain((-1.0, 0.0))
        assert subs[1] == Domain((0.0, 1.0))

    def test_intervals_three(self):
        d = Domain((0.0, 1.0, 2.0, 3.0))
        subs = list(d.intervals)
        assert len(subs) == 3
        assert subs[0] == Domain((0.0, 1.0))
        assert subs[1] == Domain((1.0, 2.0))
        assert subs[2] == Domain((2.0, 3.0))


# ======================================================================
# Affine mapping
# ======================================================================


class TestMapping:
    """Tests for forward_map, inverse_map, and map_derivative."""

    def test_forward_map_default(self):
        """On [-1,1], forward_map is the identity."""
        d = Domain((-1.0, 1.0))
        y = jnp.linspace(-1, 1, 11)
        x = d.forward_map(y)
        npt.assert_allclose(np.array(x), np.array(y), atol=1e-15)

    def test_inverse_map_default(self):
        """On [-1,1], inverse_map is the identity."""
        d = Domain((-1.0, 1.0))
        x = jnp.linspace(-1, 1, 11)
        y = d.inverse_map(x)
        npt.assert_allclose(np.array(y), np.array(x), atol=1e-15)

    def test_forward_map_scaled(self):
        """Map from [-1,1] to [0,2]."""
        d = Domain((0.0, 2.0))
        y = jnp.array([-1.0, 0.0, 1.0])
        x = d.forward_map(y)
        npt.assert_allclose(np.array(x), [0.0, 1.0, 2.0], atol=1e-15)

    def test_inverse_map_scaled(self):
        """Map from [0,2] to [-1,1]."""
        d = Domain((0.0, 2.0))
        x = jnp.array([0.0, 1.0, 2.0])
        y = d.inverse_map(x)
        npt.assert_allclose(np.array(y), [-1.0, 0.0, 1.0], atol=1e-15)

    def test_roundtrip(self):
        """forward_map(inverse_map(x)) = x for arbitrary domain."""
        d = Domain((3.0, 7.0))
        x = jnp.linspace(3.0, 7.0, 50)
        y = d.inverse_map(x)
        x_rec = d.forward_map(y)
        npt.assert_allclose(np.array(x_rec), np.array(x), rtol=1e-14, atol=1e-14)

    def test_inverse_roundtrip(self):
        """inverse_map(forward_map(y)) = y."""
        d = Domain((-5.0, 10.0))
        y = jnp.linspace(-1.0, 1.0, 50)
        x = d.forward_map(y)
        y_rec = d.inverse_map(x)
        npt.assert_allclose(np.array(y_rec), np.array(y), rtol=1e-14, atol=1e-14)

    def test_forward_map_endpoints(self):
        d = Domain((-3.0, 5.0))
        assert float(d.forward_map(jnp.array(-1.0))) == pytest.approx(-3.0, abs=1e-15)
        assert float(d.forward_map(jnp.array(1.0))) == pytest.approx(5.0, abs=1e-15)

    def test_inverse_map_endpoints(self):
        d = Domain((-3.0, 5.0))
        assert float(d.inverse_map(jnp.array(-3.0))) == pytest.approx(-1.0, abs=1e-15)
        assert float(d.inverse_map(jnp.array(5.0))) == pytest.approx(1.0, abs=1e-15)

    def test_map_derivative_default(self):
        d = Domain((-1.0, 1.0))
        assert d.map_derivative() == pytest.approx(1.0, abs=1e-15)

    def test_map_derivative_scaled(self):
        d = Domain((0.0, 2.0))
        assert d.map_derivative() == pytest.approx(1.0, abs=1e-15)

    def test_map_derivative_general(self):
        d = Domain((-3.0, 5.0))
        assert d.map_derivative() == pytest.approx(4.0, abs=1e-15)

    def test_forward_map_scalar(self):
        """forward_map works with a scalar JAX array."""
        d = Domain((0.0, 10.0))
        y = jnp.array(0.0)
        x = d.forward_map(y)
        assert float(x) == pytest.approx(5.0, abs=1e-14)

    def test_map_piecewise_error(self):
        """Mapping functions raise for piecewise domains."""
        d = Domain((-1.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="single-interval"):
            d.forward_map(jnp.array(0.0))
        with pytest.raises(ValueError, match="single-interval"):
            d.inverse_map(jnp.array(0.0))
        with pytest.raises(ValueError, match="single-interval"):
            d.map_derivative()


# ======================================================================
# Containment
# ======================================================================


class TestContainment:
    """Tests for __contains__ and is_interior."""

    def test_contains_interior(self):
        d = Domain((-1.0, 1.0))
        assert 0.0 in d
        assert 0.5 in d
        assert -0.5 in d

    def test_contains_endpoints(self):
        d = Domain((-1.0, 1.0))
        assert -1.0 in d
        assert 1.0 in d

    def test_not_contains_outside(self):
        d = Domain((-1.0, 1.0))
        assert -2.0 not in d
        assert 1.5 not in d

    def test_contains_int(self):
        d = Domain((-1.0, 1.0))
        assert 0 in d

    def test_is_interior_inside(self):
        d = Domain((-1.0, 1.0))
        assert d.is_interior(0.0)
        assert d.is_interior(0.999)
        assert d.is_interior(-0.999)

    def test_is_interior_endpoints(self):
        d = Domain((-1.0, 1.0))
        assert not d.is_interior(-1.0)
        assert not d.is_interior(1.0)

    def test_is_interior_outside(self):
        d = Domain((-1.0, 1.0))
        assert not d.is_interior(-2.0)
        assert not d.is_interior(2.0)


# ======================================================================
# Union
# ======================================================================


class TestUnion:
    """Tests for union and _merge."""

    def test_union_identical(self):
        d1 = Domain((-1.0, 1.0))
        d2 = Domain((-1.0, 1.0))
        result = d1.union(d2)
        assert result == Domain((-1.0, 1.0))

    def test_union_adds_breakpoint(self):
        d1 = Domain((-1.0, 1.0))
        d2 = Domain((-1.0, 0.0, 1.0))
        result = d1.union(d2)
        assert result == Domain((-1.0, 0.0, 1.0))

    def test_union_symmetric(self):
        d1 = Domain((-1.0, 0.0, 1.0))
        d2 = Domain((-1.0, 0.5, 1.0))
        r1 = d1.union(d2)
        r2 = d2.union(d1)
        assert r1 == r2

    def test_union_both_piecewise(self):
        d1 = Domain((-1.0, 0.0, 1.0))
        d2 = Domain((-1.0, -0.5, 0.5, 1.0))
        result = d1.union(d2)
        assert result == Domain((-1.0, -0.5, 0.0, 0.5, 1.0))

    def test_union_mismatched_support_error(self):
        d1 = Domain((-1.0, 1.0))
        d2 = Domain((0.0, 2.0))
        with pytest.raises(ValueError, match="different support"):
            d1.union(d2)


# ======================================================================
# Restrict
# ======================================================================


class TestRestrict:
    """Tests for restrict."""

    def test_restrict_to_self(self):
        d = Domain((-1.0, 1.0))
        r = d.restrict(-1.0, 1.0)
        assert r == Domain((-1.0, 1.0))

    def test_restrict_left_half(self):
        d = Domain((-1.0, 1.0))
        r = d.restrict(-1.0, 0.0)
        assert r == Domain((-1.0, 0.0))

    def test_restrict_right_half(self):
        d = Domain((-1.0, 1.0))
        r = d.restrict(0.0, 1.0)
        assert r == Domain((0.0, 1.0))

    def test_restrict_interior(self):
        d = Domain((-1.0, 1.0))
        r = d.restrict(-0.5, 0.5)
        assert r == Domain((-0.5, 0.5))

    def test_restrict_preserves_breakpoints(self):
        d = Domain((-1.0, -0.5, 0.0, 0.5, 1.0))
        r = d.restrict(-0.5, 0.5)
        assert r == Domain((-0.5, 0.0, 0.5))

    def test_restrict_partial_overlap(self):
        d = Domain((-1.0, -0.5, 0.0, 0.5, 1.0))
        r = d.restrict(-0.3, 0.3)
        assert r == Domain((-0.3, 0.0, 0.3))

    def test_restrict_error_outside(self):
        d = Domain((-1.0, 1.0))
        with pytest.raises(ValueError, match="not a sub-interval"):
            d.restrict(-2.0, 0.5)

    def test_restrict_error_reversed(self):
        d = Domain((-1.0, 1.0))
        with pytest.raises(ValueError, match="a < b"):
            d.restrict(0.5, -0.5)


# ======================================================================
# Equality
# ======================================================================


class TestEquality:
    """Tests for __eq__ and __ne__."""

    def test_equal_same(self):
        d = Domain((-1.0, 1.0))
        assert d == Domain((-1.0, 1.0))

    def test_equal_piecewise(self):
        d1 = Domain((-1.0, 0.0, 1.0))
        d2 = Domain((-1.0, 0.0, 1.0))
        assert d1 == d2

    def test_not_equal_different_bp(self):
        d1 = Domain((-1.0, 1.0))
        d2 = Domain((-1.0, 0.0, 1.0))
        assert d1 != d2

    def test_not_equal_different_endpoints(self):
        d1 = Domain((-1.0, 1.0))
        d2 = Domain((0.0, 1.0))
        assert d1 != d2

    def test_hash_equal(self):
        d1 = Domain((-1.0, 1.0))
        d2 = Domain((-1.0, 1.0))
        assert hash(d1) == hash(d2)

    def test_not_equal_to_non_domain(self):
        d = Domain((-1.0, 1.0))
        assert d != "not a domain"
        assert d != 42


# ======================================================================
# Display
# ======================================================================


class TestDisplay:
    """Tests for __repr__ and __str__."""

    def test_repr_simple(self):
        d = Domain((-1.0, 1.0))
        assert repr(d) == "Domain([-1.0, 1.0])"

    def test_repr_piecewise(self):
        d = Domain((-1.0, 0.0, 1.0))
        assert repr(d) == "Domain([-1.0, 0.0, 1.0])"

    def test_str(self):
        d = Domain((-1.0, 1.0))
        assert str(d) == "[-1.0, 1.0]"

    def test_str_piecewise(self):
        d = Domain((-1.0, 0.0, 1.0))
        assert str(d) == "[-1.0, 1.0]"


# ======================================================================
# JIT compatibility
# ======================================================================


class TestJIT:
    """Tests that forward_map and inverse_map work under jax.jit.

    JAX contract: jit=yes (domain is static via eqx.Module), vmap=yes, grad=yes.
    """

    def test_forward_map_jit(self):
        """forward_map works under JIT."""
        d = Domain((0.0, 2.0))
        y = jnp.linspace(-1, 1, 11)

        @jax.jit
        def f(y):
            return d.forward_map(y)

        result = f(y)
        expected = d.forward_map(y)
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-15)

    def test_inverse_map_jit(self):
        """inverse_map works under JIT."""
        d = Domain((0.0, 2.0))
        x = jnp.linspace(0, 2, 11)

        @jax.jit
        def f(x):
            return d.inverse_map(x)

        result = f(x)
        expected = d.inverse_map(x)
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-15)

    def test_roundtrip_jit(self):
        """Round-trip mapping under JIT."""
        d = Domain((-3.0, 7.0))

        @jax.jit
        def roundtrip(x):
            return d.forward_map(d.inverse_map(x))

        x = jnp.linspace(-3, 7, 100)
        result = roundtrip(x)
        npt.assert_allclose(np.array(result), np.array(x), rtol=1e-14, atol=1e-14)

    def test_forward_map_vmap(self):
        """forward_map works under vmap over a batch of points."""
        d = Domain((1.0, 5.0))
        y = jnp.linspace(-1, 1, 20)

        vmapped = jax.vmap(d.forward_map)
        result = vmapped(y)
        expected = d.forward_map(y)
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-15)

    def test_inverse_map_grad(self):
        """Gradient of inverse_map (should be 2/(b-a))."""
        d = Domain((1.0, 5.0))
        # d(inverse_map)/dx = 2/(b-a) = 2/4 = 0.5

        grad_fn = jax.grad(lambda x: d.inverse_map(x))
        result = grad_fn(jnp.array(3.0))
        expected = 2.0 / (5.0 - 1.0)
        assert float(result) == pytest.approx(expected, abs=1e-14)

    def test_forward_map_grad(self):
        """Gradient of forward_map (should be (b-a)/2)."""
        d = Domain((1.0, 5.0))
        # d(forward_map)/dy = (b-a)/2 = 4/2 = 2

        grad_fn = jax.grad(lambda y: d.forward_map(y))
        result = grad_fn(jnp.array(0.0))
        expected = (5.0 - 1.0) / 2.0
        assert float(result) == pytest.approx(expected, abs=1e-14)
