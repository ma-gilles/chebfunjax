"""Literal predicates from MATLAB ``tests/chebfun/test_nodots.m``.

Provenance
----------
MATLAB source : tests/chebfun/test_nodots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.matlab_expr import matlab_expression

jax.config.update("jax_enable_x64", True)

TOL = 1e-14


def _n(f):
    return float(jnp.linalg.norm(jnp.atleast_1d(jnp.asarray(f.norm(2)))))


class TestChebfunNodots:
    def test_all_matlab_assertions(self):
        x = chebfun("x")

        a = x * x
        b = chebfun(lambda t: t * t)
        c = chebfun("x*x")
        assert float((a - b).norm(2)) + float((a - c).norm(2)) < TOL  # pass(1)

        a = x / (2 + x)
        b = chebfun(lambda t: t / (2 + t))
        c = chebfun("x/(2+x)")
        assert float((a - b).norm(2)) + float((a - c).norm(2)) < TOL  # pass(2)

        a = x**2
        b = chebfun(lambda t: t**2)
        c = chebfun("x^2")
        assert float((a - b).norm(2)) + float((a - c).norm(2)) < TOL  # pass(3)

        # Native x=[x x]; matrix-style same-orientation mtimes dispatches to
        # pointwise multiplication, and the explicit times adapter is its
        # dotted-operator counterpart.
        xx = chebfun(lambda t: jnp.stack([t, t], axis=-1))
        a = xx @ xx
        b = xx.times(xx)
        assert _n(a - b) < TOL  # pass(4)

        # Python exposes pointwise division through / for Chebfun columns;
        # compare its represented result to the literal scalar formula.
        a = xx / (2 + xx)
        b = chebfun(lambda t: jnp.stack([t / (2 + t), t / (2 + t)], axis=-1))
        assert _n(a - b) < TOL  # pass(5)

        a = xx**2
        b = chebfun(lambda t: jnp.stack([t**2, t**2], axis=-1))
        assert _n(a - b) < TOL  # pass(6)

        # With vectorCheck disabled, native string x*x reaches numeric mtimes
        # with two column sample arrays and raises MATLAB:innerdim.
        with pytest.raises(ValueError, match="MATLAB:innerdim"):
            chebfun("x*x", vector_check=False)  # pass(7)

    def test_raw_mtimes_is_general_and_dotted_product_stays_elementwise(self):
        raw = matlab_expression("x*y", ("x", "y"), raw_mtimes=True)
        left = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
        right = jnp.asarray([[2.0, 0.0], [1.0, 5.0]])
        assert jnp.array_equal(raw(left, right), left @ right)
        singleton = jnp.asarray([[3.0]])
        assert jnp.array_equal(raw(singleton, right), 3 * right)
        assert jnp.array_equal(raw(left, singleton), 3 * left)

        mixed = matlab_expression("x.*y/(1+x).^2", ("x", "y"), raw_mtimes=True)
        values = jnp.asarray([0.2, 0.7])
        factors = jnp.asarray([2.0, 3.0])
        assert jnp.array_equal(
            mixed(values, factors), values * factors / (1 + values) ** 2)

        x = chebfun("x.*x", vector_check=False)
        reference = chebfun("x^2")
        assert float((x - reference).norm(2)) < TOL

    @pytest.mark.parametrize("expr", ["2*x", "x*2"])
    def test_raw_scalar_mtimes_remains_scaling(self, expr):
        actual = chebfun(expr, vector_check=False)
        reference = 2 * chebfun("x")
        assert float((actual - reference).norm(2)) < TOL
