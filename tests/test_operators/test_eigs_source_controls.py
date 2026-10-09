"""Independent source eigs controls; analytical bounds fixed before execution.

Source: @linop/eigs.m, @chebcolloc/reduce.m, @chebmatrix/chebmatrix.m,
commit7574c77680d7e82b79626300bf255498271a72df.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebpref import ChebopPref
from chebfunjax.operators import _eigs_source as source
from chebfunjax.operators.blocklinop import BlockLinop
from chebfunjax.operators.blocks import D, I, eval_at
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.operators.linop import Linop
from chebfunjax.utils.quadrature import chebpts
from tests.test_matlab_port.chebop.test_eigs_basic_matlab import (
    _source_spectrum_error,
)

# O(n*eps) allowance for bounded low-degree transforms/projections, n<=24.
# This is not a bound on ill-conditioned eigenvectors or native bit identity.
ROUNDING = 256 * jnp.finfo(jnp.float64).eps


def _dirichlet(domain, shift=0.0):
    block = -D(domain, 2) + shift*I(domain)
    return (BlockLinop(block, domain=domain)
            .add_constraint(eval_at(domain[0], domain=domain), 0.0)
            .add_constraint(eval_at(domain[-1], domain=domain), 0.0))


def test_01_projection_polynomials():
    n, order = 16, 2
    p = source.projection(n, order)
    x, y = chebpts(n+order, kind=2), chebpts(n, kind=1)
    for degree in (0, 1, 3, 15):
        assert jnp.max(jnp.abs(p @ x**degree-y**degree)) < ROUNDING


def test_02_nonunit_projected_output():
    domain = (2.0, 5.0)
    pencil = source.C2Pencil(_dirichlet(domain), 18)
    t = chebpts(18, kind=1)
    x = 3.5+1.5*t
    f = pencil.functions(2-3*x+x*x)[0]
    q = jnp.asarray([2.13, 2.97, 4.01, 4.83])
    assert jnp.max(jnp.abs(f(q)-(2-3*q+q*q))) < 32*ROUNDING
    assert tuple(f.domain.breakpoints) == domain


def test_03_complex_output():
    t = chebpts(20, kind=1)
    c = source.output_coefficients(1+2j*t+(3-4j)*t*t)
    expected = jnp.zeros(20, dtype=jnp.complex128)
    expected = expected.at[:3].set(jnp.asarray([2.5-2j, 2j, 1.5-2j]))
    assert jnp.max(jnp.abs(c-expected)) < 8*ROUNDING
    assert jnp.iscomplexobj(c)


def test_04_auto_target_unnormalized_away_from_zero():
    coarse = jnp.asarray([1., 9.])
    fine = jnp.asarray([1., 9., 20.])
    # Native minimizes raw expansion 1-norm; normalizing columns would erase
    # the deliberately distinct scales and would choose a different mode.
    coeffs = [jnp.asarray([[1., .125, 3.], [0., .125, 0.]])]
    assert source.automatic_target(coarse, fine, coeffs) == 9


def test_05_auto_target_all_changing():
    coarse, fine = jnp.asarray([2., 8.]), jnp.asarray([3., 8.1])
    assert source.automatic_target(coarse, fine, [jnp.ones((2, 2))]) == 8


def test_06_energy_filter_replacement():
    class Pencil:
        dimensions = (20,)

        @staticmethod
        def coefficients(column):
            c = jnp.zeros(20).at[0].set(1.)
            if int(column[0]) == 0:
                c = c.at[0].set(1e-12).at[-1].set(1.)
            return [[c]]

    got = source.filter_modes([0, 1, 2], jnp.asarray([[0., 1., 2.]]), 2, Pencil())
    assert jnp.array_equal(got, jnp.asarray([1, 2]))


def test_07_deflation_and_selectors():
    values = jnp.asarray([2., -3., 4j, 1000.])
    assert source.nearest(values, 'SM', 1) == [0, 1, 2]
    assert source.nearest(values, 'LR', 1) == [0, 2, 1]
    assert source.nearest(values, 'LI', 1) == [2, 0, 1]
    assert source.nearest(values, 3.9j, 1)[0] == 2
    real_spectrum = jnp.arange(1., 11.)**2
    assert _source_spectrum_error(real_spectrum) == 0
    assert _source_spectrum_error(real_spectrum.astype(jnp.complex128)) == 0
    # A genuine imaginary component must not be dropped by the adapter.
    perturbed = real_spectrum.astype(jnp.complex128).at[0].add(1e-4j)
    expected = jnp.max(jnp.abs(jnp.sqrt(-perturbed)-1j*jnp.arange(1, 11)))
    assert _source_spectrum_error(perturbed) == expected
    assert expected > 0


def test_08_preference_and_stateful_dimensions(monkeypatch):
    dimensions, checks = [], []
    actual = source.C2Pencil

    class Pencil(actual):
        def __init__(self, operator, dimension, *args, **kwargs):
            dimensions.append(tuple(operator._sizes(dimension, operator.domain)))
            super().__init__(operator, dimension, *args, **kwargs)

    def eigenvalues(pencil, k, sigma):
        n = pencil.P.shape[1]
        return jnp.asarray([1.]), jnp.ones((n, 1))

    def convergence(pencil, values, pref):
        checks.append((pref.bvpTol, pref.happinessCheck))
        return [len(checks) == 2], [1]

    monkeypatch.setattr(source, 'C2Pencil', Pencil)
    monkeypatch.setattr(source, 'eigenvalues', eigenvalues)
    monkeypatch.setattr(source, 'convergence', convergence)
    prefs = ChebopPref(minDimension=16, maxDimension=32,
                       bvpTol=2**-30, happinessCheck='strict')
    source.solve(_dirichlet((-1., 1.)), k=1, pref=prefs)
    assert dimensions == [(33,), (65,), (65,), (16,)]
    assert checks == [(2**-30, 'strict')]*2


def test_09_fixed_input_vs_equation_dimension(monkeypatch):
    dimensions = []

    def solve(operator, **kwargs):
        dimensions.append(kwargs['n'])
        f = chebfun(lambda x: x, domain=operator.domain)
        return jnp.asarray([1.]), [ChebMatrix([[f]], domain=operator.domain)]

    monkeypatch.setattr(source, 'solve', solve)
    domain = (-1., 1.)
    block = _dirichlet(domain)
    block.eigs(k=1, n=18, sigma=0)
    scalar = Linop(-D(domain, 2), domain=domain,
                   bcs=[eval_at(-1., domain=domain), eval_at(1., domain=domain)])
    scalar.eigs(k=1, n=18, sigma=0)
    assert dimensions == [18, 16]


def test_10_continuous_action_ignores_constraints():
    domain = (-1., 1.)
    scalar = Linop(-D(domain, 2), domain=domain,
                   bcs=[eval_at(-1., domain=domain), eval_at(1., domain=domain)])
    f = chebfun(lambda x: 1+x+x**3, domain=domain)
    expected = chebfun(lambda x: -6*x, domain=domain)
    assert (scalar @ f-expected).norm() < 16*ROUNDING
    matrix = scalar @ ChebMatrix([[f]], domain=domain)
    assert isinstance(matrix, ChebMatrix)
    assert (matrix[0, 0]-expected).norm() < 16*ROUNDING


def test_11_chebmatrix_continuous_diff():
    domain = (-1., 1.)
    f = chebfun(lambda x: x**3, domain=domain)
    g = chebfun(lambda x: 2*x*x, domain=domain)
    matrix = ChebMatrix([[f, g]], domain=domain)
    result = matrix.diff(2)
    expected = ChebMatrix([[chebfun(lambda x: 6*x, domain=domain),
                            chebfun(4., domain=domain)]], domain=domain)
    assert isinstance(result, ChebMatrix) and result.size == (1, 2)
    assert (result-expected).norm() < 16*ROUNDING


def test_12_numeric_diff_and_unsupported_blocks():
    matrix = ChebMatrix([[3., jnp.asarray([[1., 4., 9.]])]], domain=(-1., 1.))
    unchanged = matrix.diff(0)
    assert unchanged[0, 0] is matrix[0, 0]
    assert unchanged[0, 1] is matrix[0, 1]
    rank_zero = jnp.asarray(3.)
    assert ChebMatrix([[rank_zero]], domain=(-1., 1.)).diff(0)[0, 0] is rank_zero
    result = matrix.diff()
    assert isinstance(result, ChebMatrix)
    assert result[0, 0].size == 0
    assert jnp.array_equal(result[0, 1], jnp.asarray([[3., 5.]]))
    with pytest.raises(TypeError, match='operator block'):
        ChebMatrix([[D((-1., 1.))]], domain=(-1., 1.)).diff()


def _assert_analytic(operator, expected, *, sigma=None):
    values, columns = operator.eigs(k=len(expected), sigma=sigma)
    # Moderate modes <=5, condition-one constant-shift Sturm-Liouville spectrum:
    # eig error1e-8 is conservative relative to O(eps*N^4) differentiation error
    # at <=128, with native basic test retaining its stricter original predicate.
    assert jnp.max(jnp.abs(values-jnp.asarray(expected))) < 1e-8
    functions = ChebMatrix([[v[0, 0] for v in columns]], domain=operator.domain)
    diagonal = ChebMatrix([[complex(values[i]) if i == j else 0.
                            for j in range(len(values))] for i in range(len(values))],
                          domain=operator.domain)
    assert (operator @ functions-functions @ diagonal).norm() < 1e-7


def test_13_shifted_target_eigensystem():
    operator = _dirichlet((0., float(jnp.pi)), shift=7.)
    _assert_analytic(operator, [16., 23., 32.], sigma=23.)


def test_14_scaled_domain_eigensystem():
    operator = _dirichlet((2., 2.+2*float(jnp.pi)))
    _assert_analytic(operator, [.25, 1., 2.25])
