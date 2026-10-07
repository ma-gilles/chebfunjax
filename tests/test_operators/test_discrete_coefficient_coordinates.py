"""Independent controls for the discrete-coordinate representation adaptation.

Provenance
----------
MATLAB source: @operatorBlock/operatorBlock.m, @functionalBlock/functionalBlock.m,
@chebtech/diff.m, @chebtech2/vals2coeffs.m, @chebfun/feval.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
The manufactured identities are independent analytic controls for the explicit
Python coordinate representation adaptation; they are not MATLAB test ports.

No oracle coefficients enter the solver. The T2*T8 identity below checks the
finite-n multiplication interpolant, including aliasing, independently of LU.
The existing mandatory default cubic test must also run unchanged.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import D, I, OperatorBlock, eval_at, mult
from chebfunjax.operators.chebop import _derivative_eval_at
from chebfunjax.operators.linop import Linop, _source_coeffs2vals


@pytest.mark.parametrize('disabled', [False, True])
def test_discrete_multiplication_alias_and_noncommuting_order(disabled):
    with jax.disable_jit(disabled):
        dom = (-1., 1.)
        # T2*T8 = (T10+T6)/2; on the 9-point Cheb2 grid T10 aliases to T6.
        a = Chebfun(funs=[_Piece.from_coeffs(jnp.array([0., 0., 1.]), *dom)],
                    domain=Domain(dom))
        md = mult(a, dom)
        deriv = D(dom)
        e8 = jnp.eye(9)[:, 8]
        got = md._coordinate_fn(9) @ e8
        e6 = jnp.eye(9)[:, 6]
        assert float(jnp.max(jnp.abs(got-e6))) < 1e-13
        # (T6)' = 12*T5 + 12*T3 + 12*T1.
        expected = jnp.array([0., 12., 0., 12., 0., 12., 0., 0., 0.])
        actual = (deriv*md)._coordinate_fn(9) @ e8
        assert float(jnp.max(jnp.abs(actual-expected))) < 1e-10
        other = (md*deriv)._coordinate_fn(9) @ e8
        assert float(jnp.max(jnp.abs(actual-other))) > 1.
        # Compare the original nodal composition, not a Leibniz-expanded tree.
        values = _source_coeffs2vals(e8)
        reference = (deriv*md).matrix(9) @ values
        assert float(jnp.max(jnp.abs(_source_coeffs2vals(actual)-reference))) < 1e-10


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('degree', [3, 4])
def test_polynomial_adaptive_coordinates(degree, disabled):
    with jax.disable_jit(disabled):
        if degree == 3:
            dom = (-2., .75)
            a, b = dom
            specs = [(a, 0), (b, 1), (a, 2)]
            bc = [-13., 3*b*b+2, 6*a]
            rhs = 6.
            def exact(x):
                return x**3+2*x-1
            query = jnp.array([-1.913, -1.167, -.319, .073, .681])
        else:
            dom = (-.8, 1.3)
            a, b = dom
            specs = [(a, 0), (a, 1), (b, 0), (a, 2)]
            bc = [a**4+2*a*a-1, 4*a**3+4*a, b**4+2*b*b-1, 12*a*a+4]
            rhs = 24.
            def exact(x):
                return x**4+2*x*x-1
            query = jnp.array([-.713, -.319, .077, .691, 1.213])
        rows = [_derivative_eval_at(x, dom, k) for x, k in specs]
        op = D(dom, degree)
        solution = Linop(op, rows, dom, bc).solve(rhs)
        assert float(jnp.max(jnp.abs(solution(query)-exact(query)))) < 1e-10
        assert float((op.apply(solution)-rhs).norm()) < 1e-10
        for row, target in zip(rows, bc):
            assert float(abs(row.apply(solution)-target)) < 1e-10


@pytest.mark.parametrize('disabled', [False, True])
def test_complex_rows_and_scaling(disabled):
    with jax.disable_jit(disabled):
        dom = (-.5, 1.)
        a, b = dom
        # u=(1+2i)(x-a)^2; u''=2+4i, u(a)=u'(a)=0.
        op = (1-1j)*D(dom, 2)
        rows = [1j*eval_at(a, dom), eval_at(a, dom)*D(dom)]
        solution = Linop(op, rows, dom, [0., 0.]).solve(6+2j)
        q = jnp.array([-.37, .11, .83])
        assert float(jnp.max(jnp.abs(solution(q)-(1+2j)*(q-a)**2))) < 1e-10
        assert float((op.apply(solution)-(6+2j)).norm()) < 1e-10
        for row in rows:
            assert float(abs(row.apply(solution))) < 1e-10
        # Exact coefficient row action, including interior derivatives.
        c = jnp.array([1+2j, 2-1j, -.5+.25j, .125j])
        # p(t)=c0+c1*t+c2*(2t²-1)+c3*(4t³-3t).
        for x in (a, .13, b):
            t = (2*x-a-b)/(b-a)
            s = 2/(b-a)
            answers = [c[0]+c[1]*t+c[2]*(2*t*t-1)+c[3]*(4*t**3-3*t),
                       s*(c[1]+4*c[2]*t+c[3]*(12*t*t-3)),
                       s*s*(4*c[2]+24*c[3]*t), s**3*24*c[3]]
            for k, expected in enumerate(answers):
                f = _derivative_eval_at(x, dom, k)
                assert float(abs(f._coordinate_fn(4)@c-expected)) < 1e-10


def test_unknown_capabilities_do_not_propagate():
    dom = (-1., 1.)
    unknown = OperatorBlock(lambda disc: jnp.eye(disc.n), domain=dom)
    assert unknown._coordinate_fn is None
    assert (I(dom)+unknown)._coordinate_fn is None
    assert (D(dom)*unknown)._coordinate_fn is None
    assert (eval_at(-1., dom)*unknown)._coordinate_fn is None
    callback = mult(lambda x: 1+x, dom)
    assert callback._coordinate_fn is None


@pytest.mark.parametrize('disabled', [False, True])
def test_coordinate_assembly_never_calls_chebfun_evaluator(disabled, monkeypatch):
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
    dom = (-1., 1.)
    a = Chebfun(funs=[_Piece.from_coeffs(jnp.array([1., .25]), *dom)],
                domain=Domain(dom))
    def forbidden(*args, **kwargs):
        raise AssertionError('Coordinate assembly entered the public evaluation fast path')
    with jax.disable_jit(disabled):
        monkeypatch.setattr(Chebtech1, '__call__', forbidden)
        monkeypatch.setattr(Chebtech2, '__call__', forbidden)
        matrix = ((D(dom)*mult(a))+2*I(dom)-D(dom, 2))._coordinate_fn(12)
        row = (2*eval_at(.25, dom)-eval_at(-1., dom)*D(dom))._coordinate_fn(12)
        assert matrix.shape == (12, 12)
        assert row.shape == (12,)
        assert bool(jnp.all(jnp.isfinite(matrix)))
        assert bool(jnp.all(jnp.isfinite(row)))


@pytest.mark.parametrize('disabled', [False, True])
def test_problem_domain_mismatch_preserves_nodal_scaling(disabled):
    from chebfunjax.operators._coordinates import eligible
    with jax.disable_jit(disabled):
        op = D()  # construction [-1,1]; original matrix(disc) uses [0,1].
        problem = Linop(op, [eval_at(0., (0., 1.))], (0., 1.), [0.])
        assert not eligible(problem)
        def forbidden(n):
            raise AssertionError('Mismatched block coordinate domain was selected')
        op._coordinate_fn = forbidden
        solution = problem.solve(1.)
        q = jnp.array([.13, .41, .87])
        assert float(jnp.max(jnp.abs(solution(q)-q))) < 1e-10
        assert float((solution.diff()-1).norm()) < 1e-10
        assert float(abs(solution(0.))) < 1e-10
        bc_mismatch = Linop(D((0., 1.)), [eval_at(0.)], (0., 1.), [0.])
        assert not eligible(bc_mismatch)


@pytest.mark.parametrize('disabled', [False, True])
def test_multiplier_endpoint_overrides_match_original_diag(disabled):
    from chebfunjax.operators.linop import _source_vals2coeffs
    with jax.disable_jit(disabled):
        dom = (-2., 1.)
        f = Chebfun(funs=[_Piece.from_coeffs(jnp.array([1., .25]), *dom)],
                    domain=Domain(dom)).set_point_values(jnp.array([3+2j, -4+.5j]))
        block = mult(f, dom)
        assert block._coordinate_fn is not None
        n = 9
        values = _source_coeffs2vals(jnp.eye(n))
        expected = _source_vals2coeffs(block.matrix(n) @ values)
        actual = block._coordinate_fn(n)
        assert float(jnp.max(jnp.abs(actual-expected))) < 1e-10
        # The overridden endpoints are part of the actual nodal operator.
        diagonal = jnp.diag(block.matrix(n))
        assert float(abs(diagonal[0]-(3+2j))) < 1e-10
        assert float(abs(diagonal[-1]-(-4+.5j))) < 1e-10
        for bad in (jnp.ones((2, 1)), jnp.array([1., jnp.inf])):
            assert mult(f.set_point_values(bad), dom)._coordinate_fn is None


@pytest.mark.parametrize('disabled', [False, True])
def test_chebtech1_multiplier_noncanonical_interval(disabled):
    from chebfunjax.operators.linop import _source_vals2coeffs
    from chebfunjax.tech.chebtech import Chebtech1
    with jax.disable_jit(disabled):
        dom = (.25, 2.75)
        coeffs = jnp.array([1., -.2, .125])
        piece = _Piece(tech=Chebtech1.from_coeffs(coeffs), interval=dom)
        f = Chebfun(funs=[piece], domain=Domain(dom))
        block = mult(f, dom)
        assert block._coordinate_fn is not None
        n = 12
        values = _source_coeffs2vals(jnp.eye(n))
        expected = _source_vals2coeffs(block.matrix(n) @ values)
        assert float(jnp.max(jnp.abs(block._coordinate_fn(n)-expected))) < 1e-10


def test_finite_domain_and_derivative_capability_guards():
    from chebfunjax.operators._coordinates import derivative, evaluation, identity, valid_domain
    assert valid_domain((0., 1.))
    for domain in ((-jnp.inf, 1.), (0., jnp.inf), (jnp.nan, 1.), (1., 0.),
                   (0., 0.), (-1., 0., 1.)):
        assert not valid_domain(domain)
        assert identity(domain) is None
        assert derivative(domain, 1) is None
        assert evaluation(0., domain) is None
    for order in (-1, .5, 1., True):
        assert derivative((0., 1.), order) is None
    assert derivative((0., 1.), 0) is not None
