"""Source branch controls supplement literal native roots predicates."""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech, _trig_roots_complex
from chebfunjax.utils._polynomial_roots_source import polynomial_roots


@pytest.mark.parametrize('c', [[], [0.], [3.], [0., 0., 0.]])
def test_empty_polynomial_outputs(c):
    for prune in (False, True):
        assert _trig_roots_complex(jnp.asarray(c, dtype=jnp.complex128), prune).size == 0


@pytest.mark.parametrize('c', [[1., 2., 1.], [1e-200, 1., 0.], [0., 1., 1.],
                              [1+2j, .3-.2j, 2j]])
def test_literal_polynomial_log_without_polish(c):
    c = jnp.asarray(c, dtype=jnp.complex128)
    z = polynomial_roots(c[::-1])
    expected = (-1j/jnp.pi)*jnp.log(z.astype(jnp.complex128))
    actual = _trig_roots_complex(c, False)
    np.testing.assert_array_equal(actual, expected)


def test_strip_prunes_far_root():
    c = jnp.asarray([1e-30, 1., 0.], dtype=jnp.complex128)
    assert _trig_roots_complex(c, False).size == 1
    assert _trig_roots_complex(c, True).size == 0


@pytest.mark.parametrize('negative_zero', [False, True])
def test_principal_log_preserves_signed_zero(monkeypatch, negative_zero):
    from chebfunjax.utils import _polynomial_roots_source as poly
    z = jnp.asarray([complex(-1., -0. if negative_zero else 0.)])
    monkeypatch.setattr(poly, 'polynomial_roots', lambda _: z)
    actual = _trig_roots_complex(jnp.asarray([1., 1.]), False)
    assert float(jnp.real(actual[0])) == (-1. if negative_zero else 1.)


def test_api_flags_and_column_padding(monkeypatch):
    import chebfunjax.tech.trigtech as module
    calls = []

    def fake(c, prune=True):
        calls.append(prune)
        return jnp.asarray([1+2j] if float(jnp.real(c[0])) == 1 else [3+4j, 5+6j])

    monkeypatch.setattr(module, '_trig_roots_complex', fake)
    monkeypatch.setattr(Trigtech, 'simplify', lambda self: self)
    f = Trigtech.from_coeffs(jnp.asarray([[1., 2.], [3., 4.], [5., 6.]]))
    for flags, expected in [({'all': True}, False), ({'complex': True}, True),
                            ({'complex': True, 'all': True}, True),
                            ({'complex': True, 'prune': False}, False)]:
        r = f.roots(**flags)
        assert calls[-2:] == [expected, expected]
        np.testing.assert_array_equal(r[:, 1], [3+4j, 5+6j])
        assert complex(r[0, 0]) == 1+2j and bool(jnp.isnan(r[1, 0]))


@pytest.mark.parametrize('value', [jnp.inf, jnp.nan])
def test_nonfinite_polynomial_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        _trig_roots_complex(jnp.asarray([1., value, 1.]))
