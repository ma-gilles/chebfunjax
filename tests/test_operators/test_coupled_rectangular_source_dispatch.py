"""Independent polynomial control for source coupled linear BVP dispatch.

Chebfun7574c77 @chebop/solvebvpLinear.m -> @linop/linsolve.m.
This control supplements the unchanged original linearSystem2 predicates.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators import chebop_altdisc
from chebfunjax.operators.chebop import Chebop, SystemSolution


@pytest.mark.parametrize("backend", [None, "chebcolloc2"])
@pytest.mark.parametrize("domain", [(-1., 1.), (-1., 0., 1.)])
def test_two_sided_coupled_bvp_uses_source_rectangular_solver(monkeypatch, domain, backend):
    calls = []
    original = chebop_altdisc._solve_linear_system_ad

    def observe(*args, **kwargs):
        calls.append(args[4:9])
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("coupled differential BVP used square row replacement")

    monkeypatch.setattr(chebop_altdisc, "_solve_linear_system_ad", observe)
    monkeypatch.setattr(Chebop, "_solve_linear_system", forbidden)
    monkeypatch.setattr(Chebop, "_solve_piecewise", forbidden)
    problem = Chebop(lambda x, u, v: [u.diff()+v, v.diff()-2*u], domain)
    problem.lbc = lambda u, v: u+1
    problem.rbc = lambda u, v: v-1
    forcing = [chebfun(lambda x: 1+x*x, domain=domain), chebfun(0., domain=domain)]
    solution = problem.solve(forcing, n=12, n_min=8, n_max=32, tol=1e-11,
                             discretization=backend)
    assert isinstance(solution, SystemSolution)
    u, v = solution
    assert calls == [("chebcolloc2", 12, 8, 32, 1e-11)]
    x = jnp.linspace(-1., 1., 41)
    assert float(jnp.max(jnp.abs(u(x)-x))) < 1e-11
    assert float(jnp.max(jnp.abs(v(x)-x*x))) < 1e-11


def test_three_component_two_sided_bvp_preserves_boundary_rows():
    # Manufactured u=x, v=x^2, w=x^3, with two left constraints and one
    # right constraint. The default adaptive source schedule is retained.
    problem = Chebop(lambda x, u, v, w: [u.diff()+v, v.diff()+w,
                                       w.diff()-3*v], (-1., 1.))
    problem.lbc = lambda u, v, w: [u+1, v-1]
    problem.rbc = lambda u, v, w: w-1
    forcing = [chebfun(lambda x: 1+x*x), chebfun(lambda x: 2*x+x**3),
               chebfun(0.)]
    solution = problem.solve(forcing)
    assert isinstance(solution, SystemSolution) and len(solution) == 3
    x = jnp.linspace(-1., 1., 41)
    for component, degree in zip(solution, (1, 2, 3)):
        assert float(jnp.max(jnp.abs(component(x)-x**degree))) < 1e-11
    u, v, w = solution
    assert abs(float(u(-1.)+1)) < 1e-11
    assert abs(float(v(-1.)-1)) < 1e-11
    assert abs(float(w(1.)-1)) < 1e-11
