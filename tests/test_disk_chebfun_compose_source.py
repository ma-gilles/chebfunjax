"""Cartesian CHEBFUN composition semantics from @diskfun/compose.m (7574c77)."""
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.diskfun.diskfun import Diskfun, diskfun


@pytest.mark.parametrize("columns", [1, 2])
def test_chebfun_composition_cartesian_callback_and_values(columns, monkeypatch):
    # Independent nonradial input and asymmetric Cartesian interior, origin,
    # and boundary probes distinguish Cartesian from polar callback arguments.
    def inner(x, y):
        return -.1 + .25*x - .35*y + .15*x*y
    x = jnp.asarray([0., .17, -.43, .6, -.8, 1., 0.])
    y = jnp.asarray([0., -.31, .22, .8, -.6, 0., -1.])
    f = diskfun(inner)
    def first(t):
        return t*t + .2*t*t*t
    def second(t):
        return 1 - .4*t + t**4
    outer = cj.chebfun(first if columns == 1 else
                      lambda t: jnp.stack((first(t), second(t)), axis=-1))
    original = Diskfun.from_function
    callbacks = []
    def observe(op, *args, **kwargs):
        callbacks.append((kwargs.get("coordinates", "polar"), op(x, y)))
        return original(op, *args, **kwargs)
    monkeypatch.setattr(Diskfun, "from_function", staticmethod(observe))
    result = outer(f)
    assert len(callbacks) == columns
    for index, (coordinates, values) in enumerate(callbacks):
        assert coordinates == "cart"
        column = outer if columns == 1 else outer[index]
        assert jnp.array_equal(values, column(f.feval_cart(x, y)))
    components = [result] if columns == 1 else list(result.components)
    for component, polynomial in zip(components, [first, second]):
        assert float(jnp.max(jnp.abs(component.feval_cart(x, y)
                                     - polynomial(inner(x, y))))) < 1e-13


def test_cartesian_map_retains_tiny_radius(monkeypatch):
    # cart2pol uses hypot: ordinary squaring underflows these finite inputs.
    # Inspect the polar arguments independently of polynomial FEVAL, whose
    # own rounding can lose a tiny function value even in native Chebfun.
    monkeypatch.setattr(Diskfun, "__call__", lambda self, theta, radius: radius)
    radius = Diskfun.empty().feval_cart(jnp.asarray(3e-300), jnp.asarray(4e-300))
    assert abs(float(radius / 1e-300) - 5.) < 1e-14


def test_cartesian_allowed_exterior_is_not_clamped():
    # Native FEVAL permits r<=1+1e-8 and evaluates the continuation there.
    f = diskfun(lambda x, y: x)
    x = jnp.asarray([1., 1.+5e-9, 0., -.6])
    y = jnp.asarray([0., 0., -1.-5e-9, .8])
    assert float(jnp.max(jnp.abs(f.feval_cart(x, y)-x))) < 5e-14
    with pytest.raises(ValueError, match="pointsNotOnDisk"):
        f.feval_cart(jnp.asarray(1.+2e-8), jnp.asarray(0.))


def test_cartesian_broadcast_origin_and_nonfinite_inputs():
    f = diskfun(lambda x, y: -.1 + .25*x - .35*y + .15*x*y)
    x = jnp.asarray([[-.6], [0.], [.4]])
    y = jnp.asarray([[-.3, .1]])
    expected = -.1 + .25*x - .35*y + .15*x*y
    actual = f.feval_cart(x, y)
    assert actual.shape == (3, 2)
    assert float(jnp.max(jnp.abs(actual-expected))) < 1e-13
    origins = f.feval_cart(jnp.asarray([0., -0.]), jnp.asarray([-0., 0.]))
    assert float(jnp.max(jnp.abs(origins+.1))) < 1e-13
    assert jnp.isnan(f.feval_cart(jnp.asarray(jnp.nan), jnp.asarray(0.)))
    for x, y in [(jnp.inf, 0.), (-jnp.inf, jnp.nan), (jnp.nan, jnp.inf)]:
        with pytest.raises(ValueError, match="pointsNotOnDisk"):
            f.feval_cart(jnp.asarray(x), jnp.asarray(y))
