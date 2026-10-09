"""Disk numerical rank: shared source semantics and analytic spectra."""
import jax.numpy as jnp
import pytest

from chebfunjax.diskfun.diskfun import Diskfun


def test_empty_rank_is_empty_before_tolerance_validation():
    assert Diskfun.empty().numerical_rank([1, 2]).shape == (0,)


def test_zero_rank_is_zero_independent_of_tolerance():
    f = Diskfun.from_function(lambda theta, r: jnp.zeros_like(r))
    assert f.rank == 1
    assert f.numerical_rank() == 0
    assert f.numerical_rank([1, 2]) == 0


def test_analytic_orthogonal_disk_spectrum():
    # Radial norms squared are 1/4 and 1/72; radial cross product is
    # zero. Both angular norms squared are pi. The singular ratio is 1/4.
    f = Diskfun.from_function(lambda theta, r:
        r*jnp.cos(theta) + .75*jnp.sqrt(2.)*(r**3-2*r/3)*jnp.sin(theta))
    s = f.svd()
    assert float(jnp.max(jnp.abs(s - jnp.sqrt(jnp.pi)*jnp.asarray([.5,.125])))) < 1e-13
    assert f.numerical_rank() == 2
    assert f.numerical_rank(.2) == 2
    assert f.numerical_rank(.3) == 1
    assert f.numerical_rank(1).shape == (0,)
    assert f.numerical_rank(2).shape == (0,)


@pytest.mark.parametrize("tol, expected", [(0,2),(.25,1),(-1,3),
                                           (1,None),(float("nan"),None)])
def test_source_strict_spectral_decision(monkeypatch, tol, expected):
    # Exact spectrum isolates hard-zero, strict boundary, and last-index
    # semantics from roundoff in the existing weighted SVD algorithm.
    f = Diskfun.from_function(lambda theta, r: jnp.ones_like(r))
    monkeypatch.setattr(Diskfun, "svd", lambda self: jnp.asarray([4.,1.,0.]))
    result = f.numerical_rank(tol)
    if expected is None:
        assert result.shape == (0,)
    else:
        assert result == expected
