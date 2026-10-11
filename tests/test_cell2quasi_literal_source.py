"""Original single domain predicate and supplemental source promotion controls."""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref


def test_original_domain_predicate():
    x35 = cj.chebfun(lambda x: x, domain=[3, 5], pref=ChebfunPref())
    q = cj.cell2quasi([x35, 2])
    assert bool(jnp.all(jnp.asarray(q.cols[1].domain.breakpoints) == jnp.asarray([3, 5])))


def test_first_chebfun_template_and_complex_values():
    f = cj.chebfun(lambda x: x, domain=[3, 4, 5])
    f = f.set_point_values([[7], [8], [9]])
    q = cj.cell2quasi([2 + 3j, f, -4])
    assert q.cols[1] is f
    for i, value in [(0, 2 + 3j), (2, -4)]:
        np.testing.assert_array_equal(q.cols[i].domain.breakpoints, f.domain.breakpoints)
        np.testing.assert_array_equal(
            q.cols[i].point_values, jnp.full_like(q.cols[i].point_values, value)
        )
        np.testing.assert_array_equal(
            q.cols[i](jnp.asarray([3.0, 3.5, 4.0, 4.5, 5.0])), jnp.full((5,), value)
        )
    np.testing.assert_array_equal(f.point_values, jnp.asarray([[7], [8], [9]]))


def test_row_numeric_promotion():
    f = cj.chebfun("x", domain=[3, 5]).transpose()
    q = cj.cell2quasi([f, 2])
    assert q.is_transposed
    assert q.cols[0] is f
    np.testing.assert_array_equal(q.cols[1].domain.breakpoints, f.domain.breakpoints)


def test_source_validation_identifiers():
    import pytest

    f = cj.chebfun("x")
    for cells, identifier in [
        (f, "notacell"),
        ([], "notacheb"),
        ([2, 3], "notacheb"),
        ([cj.chebfun(lambda x: jnp.stack([x, 2 * x], axis=-1))], "notascalar"),
        ([f, cj.chebfun("x", domain=[3, 5])], "domaion"),
    ]:
        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:cell2quasi:" + identifier):
            cj.cell2quasi(cells)
