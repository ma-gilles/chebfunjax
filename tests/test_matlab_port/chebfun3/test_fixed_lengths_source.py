"""Original fixed-length constructor clauses3–7, Python keyword adapter.

Provenance
----------
MATLAB source: tests/chebfun3/test_construnctorsyntax.m; @chebfun3/constructor.m
Chebfun commit: 7574c77
The MATLAB domain/length argument orders map to Python keyword order.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3


@pytest.mark.parametrize("domain", [(-1.,1.,-1.,1.,-1.,1.), (-2.,1.,-2.,1.,-2.,1.)])
def test_native_fixed_length_predicates(domain):
    op = lambda x, y, z: jnp.sin(x*y*z)
    f = chebfun3(op, n=(10,15,5), domain=domain)
    assert f.length() == (10,15,5)
    assert f.domain == domain
    g = chebfun3(op, domain=domain, n=(10,15,5))
    assert g.length() == (10,15,5)
    assert g.domain == domain


@pytest.mark.parametrize('lengths, error', [
    ((10.5, 15, 5), TypeError), ((True, 15, 5), TypeError),
    ((0, 15, 5), ValueError), ((-1, 15, 5), ValueError),
    ((10, 15), ValueError), (10, TypeError),
])
def test_fixed_length_adapter_validation(lengths, error):
    with pytest.raises(error, match='three positive integers'):
        chebfun3(lambda x, y, z: x+y+z, n=lengths)


def test_fixed_length_samples_once_without_adaptive_fallback(monkeypatch):
    from chebfunjax.chebfun3d.chebfun3 import Chebfun3
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Fixed tensor construction must not adapt a callback')

    monkeypatch.setattr(Chebfun3, 'from_function', classmethod(forbidden))

    def op(x, y, z):
        calls.append((x.shape, y.shape, z.shape))
        return x + 2*y + 3*z

    f = chebfun3(op, n=(5, 4, 3), domain=(-2., 3., 4., 10., -7., -1.))
    assert calls == [((5, 4, 3),) * 3]
    assert abs(float(f(.2, 5., -2.)) - 4.2) < 100*jnp.finfo(jnp.float64).eps


def test_fixed_length_scalar_callback_is_not_broadcast():
    f = chebfun3(lambda x, y, z: 2., n=(5, 4, 3))
    assert f.length() == (1, 1, 1)
    # The scalar callback must take the same existing numeric scalar route.
    reference = chebfun3(2.)
    assert float(f(.2, .3, .4)) == float(reference(.2, .3, .4))
