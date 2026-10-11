"""Source controls for times and the equality predicate used by test_times.

Provenance
----------
MATLAB source : @chebfun/{times,isequal}.m, @chebtech/isequal.m,
                @trigtech/isequal.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.tech.chebtech import Chebtech2


def with_coeffs(f, c):
    return Chebfun(
        funs=[f.funs[0].with_tech(Chebtech2.from_coeffs(jnp.asarray(c)))], domain=f.domain
    )


def test_equal_requires_exact_coefficients_and_length():
    f = cj.chebfun(lambda x: 1 + 0 * x)
    a = with_coeffs(f, [1.0])
    assert a.isequal(a)
    assert not a.isequal(with_coeffs(f, [1.0 + jnp.finfo(jnp.float64).eps]))
    assert not a.isequal(with_coeffs(f, [1.0, 0.0]))


def test_breakpoint_override_uses_native_tolerance():
    f = cj.chebfun(lambda x: 1 + 0 * x)
    g = cj.chebfun(lambda x: 1 + 0 * x)
    object.__setattr__(g, "_point_values", f.point_values + 11 * jnp.finfo(jnp.float64).eps)
    assert not f.isequal(g)
    object.__setattr__(g, "_point_values", f.point_values + 9 * jnp.finfo(jnp.float64).eps)
    assert f.isequal(g)


def test_empty_type_orientation_and_domain():
    f = cj.chebfun(lambda x: x)
    assert cj.chebfun().isequal(cj.chebfun())
    assert not cj.chebfun().isequal(f)
    assert not f.isequal(1)
    assert not f.isequal(f.T)
    assert not f.isequal(cj.chebfun(lambda x: x, domain=(0, 1)))


def test_periodic_equality_checks_stored_values():
    f = cj.chebfun(jnp.cos, domain=(0, 2 * jnp.pi), periodic=True)
    g = cj.chebfun(jnp.cos, domain=(0, 2 * jnp.pi), periodic=True)
    assert f.isequal(g)
    object.__setattr__(
        g.funs[0].tech, "_values", g.funs[0].tech.values.at[0].add(jnp.finfo(jnp.float64).eps)
    )
    assert not f.isequal(g)


def test_times_rejects_orientation_and_preserves_mtimes():
    f = cj.chebfun(lambda x: x)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:times:matdim:"):
        f.times(f.T)
    assert jnp.abs(f.T * f - 2 / 3) < 10 * jnp.finfo(jnp.float64).eps
    assert f.times(f).T.isequal(f.T.times(f.T))


def test_times_thresholds_source_breakpoint_values():
    f = cj.chebfun(lambda x: 1 + 0 * x)
    object.__setattr__(f, "_point_values", jnp.asarray([1e-16, 1.0]))
    assert float(f.times(1).point_values[0]) == 0


def test_quasimatrix_equality_and_mixed_array_delegation():
    from chebfunjax.chebfun1d.linalg import Quasimatrix

    f = cj.chebfun(lambda x: jnp.stack([x, 1 + 0 * x], axis=-1))
    q = Quasimatrix(f.mat2cell(), f.domain)
    assert q.isequal(f) and f.isequal(q)
    assert q.isequal(q) and q.T.isequal(f.T)
    assert not q.isequal(q.T)
    assert not q.isequal(cj.chebfun())
    assert not q.isequal(1)
    assert not q.isequal(Quasimatrix([q.cols[0]], q.domain))
    other = Quasimatrix(
        [q.cols[0], with_coeffs(q.cols[1], [1.0 + jnp.finfo(jnp.float64).eps])], q.domain
    )
    assert not q.isequal(other)
