"""Adversarial controls for Chebfun 7574c77 @chebfun3/isequal.m."""
import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3


def test_empty_source_cases():
    empty = Chebfun3.empty()
    f = Chebfun3.from_function(lambda x, y, z: x)
    assert empty.isequal(Chebfun3.empty())
    assert not empty.isequal(f)
    assert not f.isequal(empty)


def test_equivalent_function_with_different_tucker_core_is_not_equal():
    f = Chebfun3.from_function(lambda x, y, z: x)
    # Negate one factor and the core: the represented function is unchanged,
    # but native isequal compares the stored core and factors individually.
    g = Chebfun3(cols=[-c for c in f.cols], rows=f.rows, tubes=f.tubes,
                core=-f.core, domain=f.domain)
    assert jnp.max(jnp.abs(f(jnp.asarray([-.3, .4]), 0., 0.)
                          - g(jnp.asarray([-.3, .4]), 0., 0.))) == 0
    assert not f.isequal(g)


def test_grid_alias_does_not_make_unequal_functions_equal():
    f = Chebfun3.from_function(lambda x, y, z: jnp.sin(6*jnp.pi*(x+1)))
    g = Chebfun3.from_function(lambda x, y, z: 0*x)
    grid = jnp.linspace(-1., 1., 13)
    # These functions fool the former thirteen-point-per-axis comparison.
    assert jnp.max(jnp.abs(f(grid, 0., 0.))) < 1e4*jnp.finfo(jnp.float64).eps
    assert not f.isequal(g)


def test_identical_core_still_requires_identical_factors_and_domains():
    f = Chebfun3.from_function(lambda x, y, z: x)
    same = Chebfun3(cols=f.cols, rows=f.rows, tubes=f.tubes,
                   core=f.core, domain=f.domain)
    assert f.isequal(same)
    changed_factor = Chebfun3(cols=[-c for c in f.cols], rows=f.rows,
                             tubes=f.tubes, core=f.core, domain=f.domain)
    assert not f.isequal(changed_factor)
    changed_domain = Chebfun3(cols=f.cols, rows=f.rows, tubes=f.tubes,
                             core=f.core, domain=(0., 2., -1., 1., -1., 1.))
    assert not f.isequal(changed_domain)
