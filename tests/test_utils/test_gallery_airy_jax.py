"""Source operands and JAX controls for Chebfun7574c77 Airy galleries.

MATLAB source: +cheb/gallery.m and +cheb/gallery2.m.
Native tests/cheb/test_gallery{,2}.m require construction without crashing.
The independent accuracy bound below is the existing Airy primitive gate.
"""
# uses-numpy: independent SciPy reference and assertion arrays only.
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import airy

from chebfunjax.utils.gallery import gallery
from chebfunjax.utils.gallery2 import _REGISTRY, _airycomplex, _airyreal


@pytest.mark.parametrize("name,callback,domain", [
    ("airyreal", _airyreal, (-10., 10., 0., 1.)),
    ("airycomplex", _airycomplex, (-10., 10., -5., 5.)),
])
def test_source_callback_jit_and_tangent(name, callback, domain):
    assert _REGISTRY[name][1] == domain
    a, b, c, d = domain
    x = jnp.linspace(a, b, 31)[:, None]
    y = jnp.linspace(c, d, 19)[None, :]
    z = np.asarray(x) + 1j*np.asarray(y)
    reference, derivative = airy(z)[:2]
    actual, tangent = jax.jit(lambda xx, yy: jax.jvp(
        callback, (xx, yy), (jnp.ones_like(xx), jnp.zeros_like(yy))))(x, y)
    if name == "airyreal":
        reference, derivative = reference.real, derivative.real
        assert not jnp.iscomplexobj(actual)
    else:
        assert jnp.iscomplexobj(actual)
    assert np.max(np.abs(np.asarray(actual)-reference)/np.maximum(1., np.abs(reference))) < 1e-12
    assert np.max(np.abs(np.asarray(tangent)-derivative)/np.maximum(1., np.abs(derivative))) < 1e-12
    assert np.array_equal(np.asarray(actual), np.asarray(callback(x, y)))


def test_source_real_gallery_constructs_without_scipy(monkeypatch):
    import scipy.special

    def forbidden(*args, **kwargs):
        raise AssertionError("production Airy callback called SciPy")

    monkeypatch.setattr(scipy.special, "airy", forbidden)
    f = gallery("airy")
    assert tuple(f.domain.breakpoints) == (-40., 40.)
    x = np.linspace(-40., 40., 101)
    actual = np.asarray(f(jnp.asarray(x)))
    reference = airy(x)[0]
    assert not np.iscomplexobj(actual)
    assert np.max(np.abs(actual-reference)/np.maximum(1., np.abs(reference))) < 1e-12
