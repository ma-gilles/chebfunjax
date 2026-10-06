"""Independent source real-classification and traced-complex controls.

Provenance
----------
MATLAB source : @trigtech/populate.m (nonadaptive3eps rule),
                @trigtech/vscale.m, @trigtech/feval.m
Chebfun commit: 7574c77
These are additional controls derived from literal source semantics, not
original MATLAB test assertions. All diagnostic bounds are predeclared.
JIT controls check preservation of mathematical complex input, since MATLAB
has no JAX tracing API. Their dtype adapter requires separate documentation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(jnp.finfo(jnp.float64).eps)


@pytest.mark.parametrize("scale", [1.0, 1e-40], ids=["unit", "small"])
@pytest.mark.parametrize("imaginary,real_expected", [(EPS, True), (1e-14, False)], ids=["below3eps", "above3eps"])
def test_from_coeffs_uses_source_value_scale_for_real_classification(scale, imaginary, real_expected):
    # f(x)=scale*(cos(pi*x)+i*imaginary). At the source3-point grid,
    # maxabs(values)=scale to rounding. The source3eps test distinguishes
    # the two imaginary offsets. x=0 has an independently exact result.
    coeffs = scale*jnp.asarray([0.5, 1j*imaginary, 0.5], dtype=jnp.complex128)
    f = Trigtech.from_coeffs(coeffs)
    assert f.is_real is real_expected
    actual = f(jnp.asarray(0.0))
    target_imag = 0.0 if real_expected else scale*imaginary
    assert abs(float(jnp.imag(actual))-target_imag) <= 100*EPS*abs(scale*imaginary)
    assert abs(float(jnp.real(actual))-scale) <= 100*EPS*abs(scale)


@pytest.mark.parametrize("scale", [1.0, 1e-40], ids=["unit", "small"])
@pytest.mark.parametrize("imaginary,real_expected", [(EPS, True), (1e-14, False)], ids=["below3eps", "above3eps"])
def test_from_values_uses_source3eps_real_classification(scale, imaginary, real_expected):
    values = scale*(jnp.asarray([-1.0, 0.5, 0.5])+1j*imaginary)
    f = Trigtech.from_values(values)
    assert f.is_real is real_expected
    # Checking zero/nonzero imaginary phase at x=0 avoids cancellation of
    # imaginary Fourier terms at a nontrivial phase.
    actual = f(jnp.asarray(0.0))
    target_imag = 0.0 if real_expected else scale*imaginary
    if real_expected:
        assert float(jnp.imag(actual)) == 0.0
    else:
        # Unlike the direct coefficient sum above, FFT construction mixes
        # real and imaginary components. Bound absolute arithmetic error by
        # source value scale, while still detecting a lost1e-14phase.
        assert abs(float(jnp.imag(actual))-target_imag) <= 5*EPS*abs(scale)
        assert abs(float(jnp.imag(actual))) > 0.5*abs(target_imag)


@pytest.mark.parametrize("imaginary", [1e-14, 2.0], ids=["smallcomplex", "complex"])
def test_jit_from_coeffs_retains_complex_phase(imaginary):
    @jax.jit
    def evaluate(coeffs):
        return Trigtech.from_coeffs(coeffs)(jnp.asarray(0.0))
    coeffs = jnp.asarray([0.5, 1j*imaginary, 0.5], dtype=jnp.complex128)
    result = evaluate(coeffs)
    assert jnp.issubdtype(result.dtype, jnp.complexfloating)
    assert abs(float(jnp.imag(result))-imaginary) <= 100*EPS*abs(imaginary)
    assert abs(float(jnp.real(result))-1.0) <= 100*EPS


def test_jit_from_values_retains_complex_phase():
    @jax.jit
    def evaluate(values):
        return Trigtech.from_values(values)(jnp.asarray(0.0))
    values = jnp.asarray([-1.0, 0.5, 0.5], dtype=jnp.complex128)+2j
    result = evaluate(values)
    assert abs(float(jnp.imag(result))-2.0) <= 200*EPS
    assert abs(float(jnp.real(result))-1.0) <= 100*EPS
