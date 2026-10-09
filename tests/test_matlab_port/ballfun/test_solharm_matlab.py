"""All14 literal source slots;121 real harmonics and nine complex formulas.

Provenance
----------
MATLAB source : tests/ballfun/test_solharm.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun


def test_original_slots_1_3():
    tol = 1e6 * ChebfunPref().techPrefs.chebfuneps
    max_difference = max_norm = max_laplacian = 0.
    for l in range(11):
        for m in range(-l, l + 1):
            harmonic = Ballfun.solharm(l, m)
            surface = harmonic.to_spherefun(1) / float(jnp.sqrt(2*l+3))
            reference = Spherefun.sphharm(l, m)
            max_difference = max(float((surface-reference).norm()), max_difference)
            max_norm = max(abs(float(harmonic.norm())-1), max_norm)
            max_laplacian = max(float(harmonic.laplacian().norm()), max_laplacian)
    assert max_difference < tol  # original1
    assert max_norm < tol  # original2
    assert max_laplacian < tol  # original3


@pytest.mark.parametrize("l,m,slot", [(0, 0, 4), (1, -1, 5), (1, 0, 6), (1, 1, 7),
                                     (2, -2, 8), (2, -1, 9), (2, 0, 10), (2, 1, 11), (2, 2, 12)])
def test_original_complex_slots(l, m, slot):
    tol = 1e6 * ChebfunPref().techPrefs.chebfuneps
    normalization = jnp.sqrt(2*l+3)

    def exact(r, lam, theta):
        if l == 0:
            return .5*jnp.sqrt(1/jnp.pi)*normalization
        phase = r**l * jnp.exp(1j*m*lam)
        if l == 1:
            if m == 0:
                return .5*jnp.sqrt(3/jnp.pi)*phase*jnp.cos(theta)*normalization
            sign = 1 if m < 0 else -1
            return sign*.5*jnp.sqrt(3/(2*jnp.pi))*phase*jnp.sin(theta)*normalization
        if abs(m) == 2:
            return .25*jnp.sqrt(15/(2*jnp.pi))*phase*jnp.sin(theta)**2*normalization
        if abs(m) == 1:
            sign = 1 if m < 0 else -1
            return sign*.5*jnp.sqrt(15/(2*jnp.pi))*phase*jnp.sin(theta)*jnp.cos(theta)*normalization
        return .25*jnp.sqrt(5/jnp.pi)*phase*(3*jnp.cos(theta)**2-1)*normalization

    reference = Ballfun.from_function(exact, spherical=True)
    harmonic = Ballfun.solharm(l, m, "complex")
    assert float((reference-harmonic).norm()) < tol, slot


@pytest.mark.parametrize("l,m", [(3, 2), (5, 0)])
def test_original_slots_13_14(l, m):
    tol = 1e6 * ChebfunPref().techPrefs.chebfuneps
    assert abs(float(Ballfun.solharm(l, m).norm())-1) < tol
