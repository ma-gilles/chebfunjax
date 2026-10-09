"""Independent continuous-error controls exposing fixed-grid assembly."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators._linear_altdisc import LinearDiscretization
from chebfunjax.operators.chebop import Chebop
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_oscillatory_solution_refines_beyond_old_fixed_grid(monkeypatch, backend):
    attempts = []
    original = LinearDiscretization.__init__

    def record(self, operator, dimensions, *args, **kwargs):
        attempts.append(tuple(dimensions))
        return original(self, operator, dimensions, *args, **kwargs)

    monkeypatch.setattr(LinearDiscretization, '__init__', record)
    operator = Chebop(lambda x, u, v: [u.diff()-45*v, v.diff()+45*u], [-1, 1])
    operator.lbc = lambda u, v: [u+jnp.sin(45.), v-jnp.cos(45.)]
    u, v = operator.solve(0, discretization=backend, n_max=256)
    assert (u-chebfun(lambda x: jnp.sin(45*x))).norm(jnp.inf) < 1e-9
    assert (v-chebfun(lambda x: jnp.cos(45*x))).norm(jnp.inf) < 1e-9
    expected = Chebtech1 if backend == 'chebcolloc1' else Chebtech2
    assert all(isinstance(piece.tech, expected) for entry in (u, v) for piece in entry.funs)
    assert attempts[0] == (32,)
    assert max(n for dimensions in attempts for n in dimensions) > 65


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_only_unresolved_interval_refines(monkeypatch, backend):
    attempts = []
    original = LinearDiscretization.__init__

    def record(self, operator, dimensions, *args, **kwargs):
        attempts.append(tuple(dimensions))
        return original(self, operator, dimensions, *args, **kwargs)

    monkeypatch.setattr(LinearDiscretization, '__init__', record)
    domain = [-1, 0, 1]
    operator = Chebop(lambda x, u, v: [u.diff()-v, v.diff()], domain)
    operator.lbc = lambda u, v: [u+45, v-45]
    forcing = chebfun(lambda x: jnp.where(x < 0, 0., -2025*jnp.sin(45*x)), domain=domain)
    u, v = operator.solve([0., forcing], discretization=backend, n_max=256)
    exact_u = chebfun(lambda x: jnp.where(x < 0, 45*x, jnp.sin(45*x)), domain=domain)
    exact_v = chebfun(lambda x: jnp.where(x < 0, 45., 45*jnp.cos(45*x)), domain=domain)
    assert (u-exact_u).norm(jnp.inf) < 1e-9
    assert (v-exact_v).norm(jnp.inf) < 1e-8
    assert attempts[0] == (32, 32)
    assert len(attempts) > 1
    assert all(dimensions[0] == 32 for dimensions in attempts)
    assert attempts[-1][1] > 32


@pytest.mark.parametrize('backend', ['chebcolloc1', 'ultraS'])
def test_affine_operator_and_supplied_initial_correction(backend):
    x = chebfun(lambda t: t)
    operator = Chebop(lambda t, u, v: [u.diff()-v+3, v.diff()+2], [-1, 1])
    operator.init = [2*x, chebfun(3.)]
    operator.lbc = lambda u, v: [u+1, v-1]
    u, v = operator.solve([3., 2.], n=9, discretization=backend)
    assert (u-x).norm(jnp.inf) < 1e-10
    assert (v-1).norm(jnp.inf) < 1e-10
