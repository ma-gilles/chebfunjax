"""Actual assembly shape control; no PDE or constructor replay."""
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun import _bmci, _plus
from chebfunjax.tech.trigtech import _trig_prolong_coeffs


def test_actual_shape_mixed_factor_order():
    techs = []
    for column in range(714):
        size = 1056 if column % 3 else 1055
        values = ((np.arange(size, dtype=np.float64) % 17)-8)*(column+1)/1024
        values[0], values[-1] = 0.0, -0.0
        if column % 2:
            complex_values = np.empty(size, dtype=np.complex128)
            complex_values.real = values
            complex_values.imag = -values[::-1]
            values = complex_values
        techs.append(SimpleNamespace(coeffs=jnp.asarray(values)))
    expected = np.stack([np.asarray(_trig_prolong_coeffs(t.coeffs, 1056))
                         for t in techs], axis=1)
    assert expected.shape == (1056, 714)
    for assemble in (_bmci._stack, _plus._stack):
        actual = np.asarray(assemble(techs))
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
