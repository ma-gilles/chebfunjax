"""Independent public delegation controls, native Chebfun7574c77."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun
from chebfunjax.tech.chebtech import Chebtech2


def distribution(coeffs):
    f = Bndfun(Chebtech2.from_coeffs(jnp.asarray(coeffs)), Domain((-4.0, 4.0)))
    return f, Deltafun(f, jnp.asarray([-2.0, 1.0]), jnp.asarray([[3.0, -5.0], [7.0, 11.0]]))


@pytest.mark.parametrize("shape", ["scalar", "array", "complex"])
def test_exact_padding_truncation_and_kind2(shape):
    coefficients = np.array([2.0, 4.0, 6.0, 8.0])
    if shape == "array":
        coefficients = np.column_stack([coefficients, -coefficients])
    elif shape == "complex":
        coefficients = coefficients * (1 + 2j)
    f, df = distribution(coefficients)
    np.testing.assert_array_equal(df.chebcoeffs(), coefficients)
    np.testing.assert_array_equal(df.chebcoeffs(2), coefficients[:2])
    padded = np.concatenate(
        [coefficients, np.zeros((2,) + coefficients.shape[1:], dtype=coefficients.dtype)]
    )
    np.testing.assert_array_equal(df.chebcoeffs(6), padded)
    expected = np.array([-1.0, -2.0, 3.0, 4.0])
    if shape == "array":
        expected = np.column_stack([expected, -expected])
    elif shape == "complex":
        expected = expected * (1 + 2j)
    np.testing.assert_array_equal(df.chebcoeffs(None, 2), expected)
    np.testing.assert_array_equal(df.chebcoeffs(jnp.array([]), jnp.array([])), coefficients)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBTECH:chebcoeffs:badKind"):
        df.chebcoeffs(None, 3)


def test_forwarding_preserves_argument_identity_and_result(monkeypatch):
    f, df = distribution([2.0, 4.0, 6.0, 8.0])
    sentinel = object()
    first = object()
    second = object()
    seen = []

    def provider(self, *args, **kwargs):
        seen.append((self, args, kwargs))
        return sentinel

    monkeypatch.setattr(Chebtech2, "chebcoeffs", provider)
    assert f.chebcoeffs(first, kind=second) is sentinel
    assert df.chebcoeffs(first, kind=second) is sentinel
    assert len(seen) == 2
    for instance, args, kwargs in seen:
        assert instance is f.onefun
        assert args[0] is first and kwargs["kind"] is second
