"""Actual public terminal-event construction, two and three components."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


@pytest.mark.parametrize('components', [2, 3])
def test_public_terminal_domain(components, monkeypatch):
    import chebfunjax.utils.native_ode45 as module
    original = module.native_ode45
    observed = []
    def capture(*args, **kwargs):
        sol = original(*args, **kwargs)
        observed.append(sol)
        return sol
    monkeypatch.setattr(module, 'native_ode45', capture)
    functions = [lambda x, y: 1+0*x, lambda x, y: 0*x]
    if components == 3:
        functions.append(lambda x, y: 2+0*x)
    field = Chebfun2v.from_functions(*functions, domain=(-2., 2., -2., 2.))
    T, Y = field.ode45([0., 1.], [0.]*components,
                      {'events': lambda t, y: (y[0]-.7, 1, 0)})
    sol = observed[0]
    end = float(sol['xe'][0])
    assert abs(end-.7) < 512*jnp.finfo(jnp.float64).eps
    assert Y.domain.breakpoints == (0., end)
    assert abs(float(T(end))-end) < 512*jnp.finfo(jnp.float64).eps
    if components == 2:
        assert abs(complex(Y(0.))) < 512*jnp.finfo(jnp.float64).eps
        assert abs(complex(Y(end))-.7) < 512*jnp.finfo(jnp.float64).eps
    else:
        assert bool(jnp.max(jnp.abs(Y(0.))) < 512*jnp.finfo(jnp.float64).eps)
        assert bool(jnp.max(jnp.abs(Y(end)-jnp.asarray([.7, 0., 1.4])))
                    < 512*jnp.finfo(jnp.float64).eps)
