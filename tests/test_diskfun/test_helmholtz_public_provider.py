"""Source boundary controls for @diskfun/helmholtz.m, Chebfun commit7574c77."""
import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun


def test_public_preserves_tiny_complex_shift_before_disk_projection(monkeypatch):
    # Native coeffs2diskfun projects to real values. Inspect exactly its input
    # to test the source complex coefficient solve before that deliberate map.
    monkeypatch.setattr(Diskfun, 'coeffs2diskfun', lambda c: c)
    forcing = jnp.zeros((9,8)).at[0,4].set(1.)
    actual = Diskfun.helmholtz(forcing, .7+2e-15j, lambda t:0*t, 4, 8)
    reference = Diskfun.helmholtz(forcing, .7, lambda t:0*t, 4, 8)
    assert float(jnp.max(jnp.abs(jnp.imag(reference)))) == 0
    assert float(jnp.max(jnp.abs(jnp.imag(actual)))) > 0


def test_public_poisson_does_not_call_numpy_solve(monkeypatch):
    import importlib
    module = importlib.import_module('chebfunjax.diskfun.diskfun')
    def forbidden(*args, **kwargs):
        raise AssertionError('disk Helmholtz delegated its solve to NumPy')
    monkeypatch.setattr(module.np.linalg, 'solve', forbidden)
    actual = Diskfun.poisson(lambda t,r: 4+0*t*r, lambda t: 0*t, 10, 12)
    exact = Diskfun.from_function(lambda t,r: r*r-1+0*t)
    assert float((actual-exact).norm()) < 1e-12
