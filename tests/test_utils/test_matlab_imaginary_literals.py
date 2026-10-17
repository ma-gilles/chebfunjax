"""Authored imaginary suffix controls; original roots18 remains separate.

Provenance: MATLAB numeric literal grammar, test_roots.m predicate18,
Chebfun7574c77. Tests inspect parsing and JAX evaluation, not root accuracy.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.utils.matlab_expr import matlab_expression


@pytest.mark.parametrize('literal,value', [('1i',1j),('.5i',.5j),('2e-3i',.002j),
                                          ('3j',3j),('4.i',4j)])
def test_numeric_imaginary_literal(literal,value):
    f = matlab_expression(literal+'+x', ('x',))
    x = jnp.asarray([-.3,.2])
    np.testing.assert_array_equal(f(x),value+x)
    np.testing.assert_array_equal(jax.jit(f)(x),value+x)


def test_suffix_does_not_rewrite_identifier_or_variable_i():
    f = matlab_expression('xi+1i', ('xi',))
    np.testing.assert_array_equal(f(jnp.asarray([1.,2.])), [1+1j,2+1j])
    f = matlab_expression('i+1i', ('i',))
    np.testing.assert_array_equal(f(jnp.asarray([1.,2.])), [1+1j,2+1j])


def test_public_string_infers_variable_after_imaginary_literal():
    f = cj.chebfun('2i+z')
    np.testing.assert_allclose(f(jnp.asarray([-.3,.2])),[-.3+2j,.2+2j],atol=1e-14,rtol=0)
