"""Independent Decimal magnitude controls for the JAX logarithm adapter.

Provenance
----------
Analytic identity Re(log(z))=log(Re(z)^2+Im(z)^2)/2, evaluated at 500
Decimal digits from exact input floats. Supplemental 8eps scaled comparisons
follow the existing boundary suite; no native acceptance bound is changed.
"""
from decimal import Decimal, localcontext

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS


@pytest.mark.parametrize('method', ['log', 'log1p', 'log2', 'log10'])
@pytest.mark.parametrize('real, imag', [(.6, .8), (-.7, .31),
                                       (1e200, -1e200), (1e-200, 1e-200)])
def test_magnitude_decimal_reference(method, real, imag):
    with localcontext() as context:
        context.prec = 500
        x, y = Decimal.from_float(real), Decimal.from_float(imag)
        if method == 'log1p':
            x += 1
        expected = (x*x+y*y).ln()/2
        if method in ('log2', 'log10'):
            expected /= Decimal(2 if method == 'log2' else 10).ln()
        expected = float(expected)
    actual = complex(_UNARY_FNS[method](jnp.asarray(complex(real, imag)))).real
    assert abs(actual-expected) <= 8*np.finfo(float).eps*max(1., abs(expected))


@pytest.mark.parametrize('method, real', [('log', 1.), ('log1p', 0.),
                                         ('log1p', 2.**-60), ('log1p', -2.**-60)])
def test_tiny_magnitude_retained_near_unit_circle(method, real):
    imag = 2.**-40
    with localcontext() as context:
        context.prec = 500
        x, y = Decimal.from_float(real), Decimal.from_float(imag)
        if method == 'log1p':
            x += 1
        expected = float((x*x+y*y).ln()/2)
    actual = complex(_UNARY_FNS[method](jnp.asarray(complex(real, imag)))).real
    assert actual != 0
    assert abs(actual-expected) <= 8*np.finfo(float).eps*abs(expected)
