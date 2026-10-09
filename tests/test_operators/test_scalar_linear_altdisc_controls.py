"""Independent exact scalar AD, boundary and backend-routing controls."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_scalar_linear_requested_backend_and_initial_correction(monkeypatch, backend):
    import chebfunjax.operators.chebop_altdisc as module

    def forbidden(*args, **kwargs):
        raise AssertionError('Exact scalar linear route must not perturb the operator')
    monkeypatch.setattr(module, '_frechet_blocks', forbidden)
    x = chebfun(lambda t: t)
    operator = Chebop(lambda t, u: u.diff(2)+.01*t*u, [-1, 1])
    operator.init = 2*x+3
    operator.lbc = operator.rbc = 1.
    original = operator.solve
    calls = []
    def recorded(*args, **kwargs):
        calls.append(kwargs.get('discretization'))
        return original(*args, **kwargs)
    monkeypatch.setattr(operator, 'solve', recorded)
    solution = operator.solve(2+.01*x**3, n=8, discretization=backend)
    assert calls == [backend]
    assert (solution-x**2).norm(jnp.inf) < 1e-10
    assert jnp.abs(solution(-1)-1) < 1e-10
    assert jnp.abs(solution(1)-1) < 1e-10


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_exact_nonlocal_boundary_functionals(backend):
    operator = Chebop(lambda x, u: u.diff(2), [-1, 1])
    operator.bc = lambda x, u: [u.diff()(0)-1, u.sum()]
    solution = operator.solve(0, n=8, discretization=backend)
    x = chebfun(lambda t: t)
    assert (solution-x).norm(jnp.inf) < 1e-10
    assert jnp.abs(solution.diff()(0)-1) < 1e-10
    assert jnp.abs(solution.sum()) < 1e-10


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_nonlocal_equation_backend_capability(monkeypatch, backend):
    operator = Chebop(lambda u: u.diff(2)+u.sum(), [-1, 1])
    operator.lbc = operator.rbc = 0.
    original = operator.solve
    def recorded(*args, **kwargs):
        assert kwargs.get('discretization') == backend, 'No default seed for nonlocal equation'
        return original(*args, **kwargs)
    monkeypatch.setattr(operator, 'solve', recorded)
    if backend == 'ultraS':
        # Native coeffsDiscretization has no conversion for this operator.
        with pytest.raises(TypeError, match='COEFFSDISCRETIZATION:instantiate:fail'):
            operator.solve(1, n=8, discretization=backend)
    else:
        solution = operator.solve(1, n=8, discretization=backend)
        exact = chebfun(lambda x: 1.5*(x*x-1))
        assert (solution-exact).norm(jnp.inf) < 1e-10
        assert jnp.abs(solution(-1.)) < 1e-10
        assert jnp.abs(solution(1.)) < 1e-10


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_fifth_derivative_and_numeric_derivative_boundary_vector(backend):
    operator = Chebop(lambda u: u.diff(5)+1e-9*u.diff(2), [-1, 1])
    operator.lbc = [-1., 5., -20., 60., -120.]
    x = chebfun(lambda t: t)
    solution = operator.solve(120+20e-9*x**3, n=8, discretization=backend)
    assert (solution-x**5).norm(jnp.inf) < 1e-8
