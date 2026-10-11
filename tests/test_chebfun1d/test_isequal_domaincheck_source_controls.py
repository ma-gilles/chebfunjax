"""Native captured numeric/NaN/equality domain controls, Chebfun7574c77.

Provenance
----------
MATLAB source : @chebfun/{isequal,domainCheck,hscale}.m,
                @classicfun/isequal.m, @chebtech/isequal.m
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.tech.chebtech import Chebtech2

EPS = float(jnp.finfo(jnp.float64).eps)


def test_numeric_empty_and_nonempty_source_dispatch():
    e = cj.chebfun()
    f = cj.chebfun(jnp.sin)
    assert e.isequal([]) and e.isequal(jnp.empty((2, 0)))
    assert not e.isequal(jnp.nan)
    assert not f.isequal(jnp.asarray([0.0, 1.0]))
    assert not f.isequal(jnp.asarray([jnp.nan, jnp.nan]))
    with pytest.raises(TypeError, match="MATLAB:structRefFromNonStruct:"):
        f.isequal(jnp.asarray([-1.0, 1.0]))


def test_domain_check_tolerance_then_exact_fun_domains():
    f = cj.chebfun(jnp.sin)
    assert f.domain_check(jnp.asarray([-1 + EPS, 1 + EPS]))
    assert not f.domain_check(jnp.asarray([-1 + 20 * EPS, 1 + 20 * EPS]))
    g = cj.chebfun(jnp.sin, domain=(-1 + EPS, 1 + EPS))
    assert f.domain_check(g)
    assert not f.isequal(g)


def test_pointvalue_nan_and_inf_follow_native_comparison():
    f = cj.chebfun(jnp.sin)
    g = cj.chebfun(jnp.sin)
    object.__setattr__(g, "_point_values", jnp.asarray([jnp.nan, 1000.0]))
    assert f.isequal(g)
    object.__setattr__(f, "_point_values", jnp.asarray([jnp.inf, jnp.inf]))
    object.__setattr__(g, "_point_values", jnp.asarray([jnp.inf, jnp.inf]))
    assert f.isequal(g)


def test_nan_coefficients_are_not_equal_even_to_self():
    f = cj.chebfun(jnp.sin)
    tech = Chebtech2.from_coeffs(jnp.asarray([jnp.nan]))
    g = Chebfun(funs=[f.funs[0].with_tech(tech)], domain=f.domain)
    assert not g.isequal(g)


def test_unbounded_domaincheck_distinguishes_numeric_endpoints():
    f = cj.chebfun(lambda x: jnp.exp(-(x**2)), domain=(-jnp.inf, jnp.inf))
    assert f.domain_check(f)
    assert not f.domain_check(jnp.asarray([-jnp.inf, jnp.inf]))
    assert cj.chebfun().domain_check([])
    assert not f.domain_check([])


def test_array_quasimatrix_numeric_dispatch_and_outer_domains():
    from chebfunjax.chebfun1d.linalg import Quasimatrix

    f = cj.chebfun(lambda x: jnp.stack([x, x**2], axis=-1))
    q = Quasimatrix(f.mat2cell(), f.domain)
    assert f.domain_check(q) and f.domain_check(jnp.asarray([-1.0, 1.0]))
    assert not q.isequal([]) and not q.isequal(jnp.asarray([jnp.nan, jnp.nan]))
    with pytest.raises(TypeError, match="MATLAB:structRefFromNonStruct:"):
        q.isequal(jnp.asarray([-1.0, 1.0]))
    with pytest.raises(TypeError, match="MATLAB:structRefFromNonStruct:"):
        f.isequal(jnp.asarray([-1.0, 1.0]))
    broken = cj.chebfun(jnp.sin, domain=(-1.0, 0.0, 1.0))
    smooth = cj.chebfun(jnp.sin)
    assert broken.domain_check(smooth)
    assert not broken.isequal(smooth)


def test_bounded_hscale_has_no_unit_floor():
    f = cj.chebfun(jnp.sin, domain=(0.0, 0.1))
    near = jnp.nextafter(jnp.float64(0.1), jnp.inf)
    assert f.domain_check(jnp.asarray([0.0, near]))
    assert not f.domain_check(jnp.asarray([0.0, 0.1 + 2 * EPS]))


def test_numeric_row_column_and_matrix_shape_match_native_capture():
    f = cj.chebfun(jnp.sin)
    column = jnp.asarray([[-1.0], [1.0]])
    matrix = jnp.asarray([[-1.0, 0.0], [0.0, 1.0]])
    assert not f.domain_check(column) and not f.isequal(column)
    assert f.domain_check(matrix)
    with pytest.raises(TypeError, match="MATLAB:structRefFromNonStruct:"):
        f.isequal(matrix)
    empty = cj.chebfun()
    assert empty.isequal("") and empty.domain_check("")
    # On a very short domain the expanded matrix norm can be below tolerance.
    near = cj.chebfun(1.0, domain=(1.0, 1.0 + EPS))
    assert near.domain_check(jnp.asarray([[1.0], [1.0 + EPS]]))
