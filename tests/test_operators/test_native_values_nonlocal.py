"""Native C1 nonlocal stack controls, independent of source solve predicates.

Provenance
----------
MATLAB source: @valsDiscretization/instantiate.m, @functionalBlock/functionalBlock.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators._linear_altdisc import solve_operator
from chebfunjax.operators._native_values import FirstKindDisc
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.blocks import D, OperatorBlock, diag, eval_at, sum_functional
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.diffmat import diffmat
from chebfunjax.utils.quadrature import chebpts, chebweights


def test_complex_polynomial_nondefault_interval():
    domain = (1., 4.)
    disc = FirstKindDisc((8,), domain)
    block = D(domain, 2)+sum_functional(domain)
    values = (1+2j)*disc.points**2-2*disc.points
    actual = block._values_capability.realize(disc) @ values
    assert jnp.max(jnp.abs(actual-(8+46j))) < 1e-11


def test_underresolved_stack_preserves_first_kind_multiplication():
    domain = (-1., 1.)
    disc = FirstKindDisc((8,), domain)
    multiplier = chebfun(lambda x: x**9, domain=domain)
    block = D(domain)*diag(multiplier)+sum_functional(domain)
    x1, x2 = chebpts(8, kind=1), chebpts(8, kind=2)
    expected = (diffmat(8, 1, kind=1) @ jnp.diag(x1**9)
                + jnp.tile(chebweights(8, kind=1), (8, 1)))
    actual = block._values_capability.realize(disc)
    assert jnp.max(jnp.abs(actual-expected)) < 1e-11
    # A coordinate transfer of a C2 stack keeps different intermediate aliasing.
    c2 = diffmat(8, 1) @ jnp.diag(x2**9)+jnp.tile(chebweights(8), (8, 1))
    to_c2 = Chebtech2.coeffs2vals(Chebtech1.vals2coeffs(jnp.eye(8)))
    to_c1 = Chebtech1.coeffs2vals(Chebtech2.vals2coeffs(jnp.eye(8)))
    assert jnp.max(jnp.abs((actual-to_c1 @ c2 @ to_c2) @ x1**5)) > 1


def test_integral_and_directional_evaluation_couple_unequal_pieces():
    domain = (-2., 0., 3.)
    disc = FirstKindDisc((6, 8), domain)
    values = jnp.concatenate([jnp.full(6, 1+2j), jnp.full(8, 3-1j)])
    total = sum_functional(domain).promote()._values_capability.realize(disc) @ values
    assert jnp.max(jnp.abs(total-(11+1j))) < 1e-12
    for direction, expected in [(-1, 1+2j), (1, 3-1j), (0, 3-1j)]:
        promoted = eval_at(0., domain, direction).promote()
        assert jnp.max(jnp.abs(promoted._values_capability.realize(disc) @ values-expected)) < 1e-12


def test_capability_domain_preserves_hidden_coefficient_breakpoint():
    domain = (-1., 1.)
    coefficient = chebfun(lambda x: abs(x-.2), domain=(-1., .2, 1.))
    block = D(domain, 2)+diag(coefficient, domain=domain)*sum_functional(domain)
    L = BlockLinop(block, domain=domain)
    L = L.add_constraint(eval_at(-1., domain), -1.)
    L = L.add_constraint(eval_at(1., domain), 1.)
    solution, disc, _ = solve_operator(L, [0.], backend='chebcolloc1', n=8)
    assert disc.domain == (-1., .2, 1.)
    assert jnp.max(jnp.abs(disc.A[-8:, :10])) > .01
    assert (solution[0]-chebfun(lambda x: x, domain=domain)).norm(jnp.inf) < 1e-10
    assert all(isinstance(piece.tech, Chebtech1) for piece in solution[0].funs)


def test_missing_capability_is_explicit_without_c2_fallback():
    def forbidden(_disc):
        raise AssertionError('C2 fallback must not be evaluated')
    unknown = OperatorBlock(forbidden)
    with pytest.raises(TypeError, match='No native first-kind stack capability'):
        solve_operator(BlockLinop(unknown), [0.], backend='chebcolloc1', n=8)


def test_broken_differential_coefficients_are_not_caught_as_missing_capability():
    def broken():
        raise TypeError('coefficient sentinel')
    block = OperatorBlock(lambda disc: jnp.eye(disc.n), coeff_fn=broken)
    with pytest.raises(TypeError, match='coefficient sentinel'):
        solve_operator(BlockLinop(block), [0.], backend='chebcolloc1', n=8)


def test_complex_multiplier_matrix_preserves_cross_interval_coupling():
    domain = (-1., 0., 2.)
    coefficient = chebfun(lambda x: 1+2j*x, domain=domain)
    disc = FirstKindDisc((6, 8), domain)
    block = D(domain, 2)+diag(coefficient)*sum_functional(domain)
    matrix = block._values_capability.realize(disc)
    assert jnp.iscomplexobj(matrix)
    # Integral of x² over [-1,2] is3; coefficient multiplication is complex.
    expected = 2+3*(1+2j*disc.points)
    assert jnp.max(jnp.abs(matrix @ disc.points**2-expected)) < 1e-11
    assert jnp.max(jnp.abs(matrix[:6, 6:])) > .1
