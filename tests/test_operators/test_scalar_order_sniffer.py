"""Scalar composition orders and source adaptive eligibility.

The source Krylov divergence operator differentiates the flux, so its
order is two even though each individual diff call has default order one.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import ChebColloc2Disc
from chebfunjax.operators.chebop import Chebop, _OrderSniffer


@pytest.mark.parametrize("expression,order", [
    (lambda u: u.diff().diff(), 2),
    (lambda u: u.diff(2).diff(3), 5),
    (lambda u: (u.diff(2) + u.diff()).diff(3), 5),
    (lambda u: (u.diff() + u.diff(2)).diff(3), 5),
    (lambda u: u.diff(2) + u.diff(), 2),
    (lambda u: u.diff().sin().diff(), 2),
])
def test_composed_expression_order(expression, order):
    u = _OrderSniffer()
    expression(u)
    assert u.order == order


def test_independent_branches_keep_their_own_depth():
    u = _OrderSniffer()
    high = u.diff(3)
    low = u.diff()
    low.diff()
    assert u.order == 3
    (low + high).diff(2)
    assert u.order == 5
    u.diff()
    assert u.order == 5


def test_original_krylov_operator_metadata():
    # Literal source operands and boundary values, without solving.
    x = Chebfun.identity(Domain((-1., 1.)))
    a = 2 + (70 * jnp.pi * x).cos()
    L = Chebop(lambda u: -(a * u.diff()).diff() + (1 + x**12) * u,
               (-1., 1.), 3., -5.)
    assert L._sniff_order(x, 8) == L._op_order() == 2
    linop = L._build_linop()
    assert linop.L.order == 2
    assert linop._source_adaptive_eligible()


def test_divergence_assembly_matches_independent_polynomial_action():
    # Analytic derivative of -( (2+x^2) u' )' + (1+x)u.
    # A cubic exact polynomial avoids solving or fitting reference output.
    L = Chebop(lambda x, u: -((2 + x**2) * u.diff()).diff() + (1+x)*u,
               (-1., 1.), 0., 0.)
    disc = ChebColloc2Disc(24, (-1., 1.))
    from chebfunjax.utils.quadrature import chebpts
    x = np.asarray(chebpts(24, kind=2))
    values = 1 + 2*x - x**2 + 3*x**3
    first = 2 - 2*x + 9*x**2
    second = -2 + 18*x
    expected = -(2*x*first + (2+x**2)*second) + (1+x)*values
    actual = np.asarray(L._linearize_op().matrix(disc)) @ values
    assert np.max(np.abs(actual-expected)) / np.max(np.abs(expected)) < 1e-9


def test_composed_order_respects_original_maximum_and_nonlocal_policy():
    x = Chebfun.identity(Domain((-1., 1.)))
    L = Chebop(lambda u: u.diff(2).diff(3), (-1., 1.))
    assert L._sniff_order(x, 5) == 5
    assert L._sniff_order(x, 4) is None
    nonlocal_op = Chebop(lambda u: u.diff().diff() + u.cumsum(), (-1., 1.))
    assert nonlocal_op._sniff_order(x, 8) is None
