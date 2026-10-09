"""Independent exact-AD, source seed, fitBC and native C2 controls."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators._coupled_newton import fit_bcs, linearize
from chebfunjax.operators._linear_altdisc import solve_operator
from chebfunjax.operators.chebop import Chebop


def _problem():
    N = Chebop(lambda x, u, v: [u.diff(), v.diff()+v*u.sum()+u(.3)])
    N.bc = lambda x, u, v: [u(0)-1, u(0)*v(.5)]
    return N


def test_public_zero_seed_retains_zero_constraint_and_flags():
    N = _problem()
    N.init = [3, 4]
    L, residual, flags = N.linearize()
    assert all(r.norm(jnp.inf) == 0 for r in residual)
    assert len(L.constraint) == 2
    assert all(b.iszero for b in L.constraint[1][0])
    assert flags[3] is True  # native post-product zero-Jacobian classification
    assert L.matrix(3).shape == (8, 8)
    assert bool(jnp.all(L.matrix(3)[1] == 0))


def test_nonzero_exact_product_boundary_and_equation_action():
    N = _problem()
    u = chebfun(lambda x: 1+x*x)
    v = chebfun(lambda x: 2-x)
    h = chebfun(lambda x: 3+x)
    k = chebfun(lambda x: x*x)
    L, _, _ = N.linearize([u, v])
    actual = L.A.blocks[1][0].apply(h)+L.A.blocks[1][1].apply(k)
    expected = k.diff()+k*u.sum()+v*h.sum()+h(.3)
    assert (actual-expected).norm(jnp.inf) < 2e-12
    row = L.constraint[1][0]
    assert abs(row[0].apply(h)+row[1].apply(k)
               -(h(0)*v(.5)+u(0)*k(.5))) < 2e-13
    assert abs(L.constraint[1][1]-u(0)*v(.5)) < 2e-13


def test_source_fit_removes_trivial_rows_locally():
    N = _problem()
    L, _, _ = linearize(N)
    fitted = fit_bcs(L)
    assert (fitted[0]-1).norm(jnp.inf) < 2e-13
    assert fitted[1].norm(jnp.inf) == 0
    assert len(L.constraint) == 2
    assert all(b.iszero for b in L.constraint[1][0])
    fresh = _problem()
    assert fresh.init is None


def test_rank_deficient_fit_uses_native_warning_and_zero():
    N = Chebop(lambda x, u, v: [u.diff(), v.diff()])
    N.bc = lambda x, u, v: [u(0)-1, u(0)-2]
    L, _, _ = linearize(N)
    with pytest.warns(RuntimeWarning, match='CHEBFUN:LINOP:fitBCs:failure'):
        fitted = fit_bcs(L)
    assert all(u.norm(jnp.inf) == 0 for u in fitted)


def test_complex_piecewise_second_kind_rhs_projection_recovery():
    domain = (-1., 0., 1.)
    N = Chebop(lambda x, u, v: [u.diff()+v.sum(), v.diff()], domain=domain)
    N.bc = lambda x, u, v: [u(-1), v(1)]
    exact = [chebfun(lambda x: (1+1j)*(x+1), domain=domain),
             chebfun(lambda x: (2-1j)*(x-1), domain=domain)]
    L, _, _ = linearize(N)
    answer, disc, happy = solve_operator(L, N(*exact), backend='chebcolloc2', n=5)
    assert happy and disc.dimensions == (5, 5)
    assert disc.A.shape == (24, 24)
    assert all((a-b).norm(jnp.inf) < 2e-12 for a, b in zip(answer, exact))
    assert abs(answer[0](-1)) < 2e-13
    assert abs(answer[1](1)) < 2e-13
    assert not disc.is_factored  # native predicate uses equation sizes only


def test_nonlocal_ultras_rejected_without_backend_substitution():
    N = _problem()
    L, _, _ = linearize(N, [1, 0])
    with pytest.raises(TypeError, match='COEFFSDISCRETIZATION:instantiate:fail'):
        solve_operator(L, [0, 0], backend='ultraS', n=5)


def test_tiny_nonzero_fit_row_is_not_removed():
    N = Chebop(lambda x, u, v: [u.diff(), v.diff()])
    N.bc = lambda x, u, v: [1e-100*(u(0)-1)]
    L, _, _ = linearize(N)
    fitted = fit_bcs(L)
    assert (fitted[0]-1).norm(jnp.inf) < 2e-13
    assert fitted[1].norm(jnp.inf) == 0


def test_fifth_full_rank_fit_still_returns_zero(monkeypatch):
    # Two nontrivial equal rows stay rank deficient until a fifth derivative
    # becomes representable on size6 (the fifth dimension attempt).
    from chebfunjax.operators.blocks import FunctionalBlock
    N = Chebop(lambda x, u, v: [u.diff(5), v.diff()])
    N.bc = lambda x, u, v: [u(0)-1, u(0)+u.diff(5)(0)-2]
    L, _, _ = linearize(N)
    target = L.constraint[1][0][0]
    base = L.constraint[0][0][0]
    original = FunctionalBlock.matrix
    seen = []
    def matrix(self, disc):
        if self is target:
            seen.append(disc.n)
            if disc.n < 6:
                return original(base, disc)
        return original(self, disc)
    monkeypatch.setattr(FunctionalBlock, 'matrix', matrix)
    with pytest.warns(RuntimeWarning, match='CHEBFUN:LINOP:fitBCs:failure'):
        fitted = fit_bcs(L)
    assert seen == [2, 3, 4, 5, 6]
    assert all(u.norm(jnp.inf) == 0 for u in fitted)


def test_c2_hidden_nonlocal_coefficient_breakpoint():
    from chebfunjax.operators.blocklinop import BlockLinop
    from chebfunjax.operators.blocks import D, diag, eval_at, sum_functional
    domain = (-1., 1.)
    coefficient = chebfun(lambda x: abs(x-.2), domain=(-1., .2, 1.))
    L = BlockLinop(D(domain, 2)+diag(coefficient, domain)*sum_functional(domain))
    L = L.add_constraint(eval_at(-1., domain), -1.)
    L = L.add_constraint(eval_at(1., domain), 1.)
    solution, disc, _ = solve_operator(L, [0.], backend='chebcolloc2', n=5)
    assert disc.domain == (-1., .2, 1.)
    assert jnp.max(jnp.abs(disc.A[-5:, :7])) > .01
    assert (solution[0]-chebfun(lambda x: x)).norm(jnp.inf) < 1e-10


def test_numeric_zero_metadata_preserves_native_operator_functional_distinction():
    from chebfunjax.autodiff.adchebfun import _scale_jacobian
    from chebfunjax.operators.blocks import D, eval_at
    derivative = D((-1., 1.), 2)
    functional = eval_at(0., (-1., 1.))*derivative
    op = _scale_jacobian(derivative, jnp.asarray(0.))
    row = _scale_jacobian(functional, jnp.asarray(0.))
    assert op.iszero and op.order == 0
    assert row.iszero and row.order == functional.order
    assert op.isnotdiffint == derivative.isnotdiffint
    assert row.isnotdiffint == functional.isnotdiffint
    assert op._values_capability is not None and row._values_capability is not None
    assert jnp.max(jnp.abs(op.matrix(5))) == 0
    assert jnp.max(jnp.abs(row.matrix(5))) == 0


def test_dynamic_scalar_cannot_invent_static_block_metadata():
    import jax

    from chebfunjax.autodiff.adchebfun import _scale_jacobian
    from chebfunjax.operators.blocks import D
    with pytest.raises(jax.errors.TracerBoolConversionError):
        jax.jit(lambda factor: _scale_jacobian(D(), factor).matrix(3))(1.)


def test_native_zero_function_vs_zero_numeric_linearity():
    first = Chebop(lambda x, u, v: [u.diff(), v.diff()])
    first.bc = lambda x, u, v: [u(0)-1, u(0)*v(.5)]
    first.init = [1, 0]
    _, _, public_flags = first.linearize()
    _, _, solve_flags = linearize(first, first.init)
    second = _problem()
    _, _, second_flags = linearize(second)
    assert all(public_flags)  # publiczero native functional-product quirk
    assert not solve_flags[3]  # solve explicitly uses first.init
    assert second.init is None
    assert not second_flags[0] and second_flags[3]


def test_public_matrix_constraint_action():
    from chebfunjax.utils.quadrature import chebpts
    N = _problem()
    u, v = chebfun(lambda x: 1+x*x), chebfun(lambda x: 2-x)
    h, k = chebfun(lambda x: 3+x), chebfun(lambda x: x*x)
    L, _, _ = N.linearize([u, v])
    points = chebpts(4, kind=2)
    actual = L.matrix(3) @ jnp.concatenate([h(points), k(points)])
    expected = jnp.concatenate([jnp.asarray([h(0), h(0)*v(.5)+u(0)*k(.5)]),
                                h.diff()(chebpts(3, kind=1)),
                                (k.diff()+k*u.sum()+v*h.sum()+h(.3))(chebpts(3, kind=1))])
    assert jnp.max(jnp.abs(actual-expected)) < 2e-12
