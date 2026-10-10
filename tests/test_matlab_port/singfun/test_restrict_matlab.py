"""All six original restriction spot checks, including every subinterval.

Provenance
----------
MATLAB source: tests/singfun/test_restrict.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.fun.singfun import Singfun

A, B, C, D = .64, -.64, 1.28, -1.28
CASES = [
    (lambda x: (1+x)**A*jnp.exp(x), (A, 0.), [-.2, .1], (0, 0)),
    (lambda x: (1+x)**D*jnp.sin(50*jnp.pi*x), (D+1, 0.), [-1., .3], (1, 0)),
    (lambda x: (1-x)**C*jnp.cos(x), (0., C), [-1., -.7, 1.], (0, 0)),
    (lambda x: (1-x)**B, (0., B), [-.9, -.3, .7, 1.], (0, 1)),
    (lambda x: (1+x)**B*jnp.sin(x)*(1-x)**D, (B, D), [-1., -.9, .5, .7, 1.], (1, 1)),
    (lambda x: (1+x)**B*jnp.sin(x)*(1-x)**(3*C), (B, B), [-1+1e-4, 1-1e-4], (0, 0)),
]


@pytest.mark.parametrize('operator,exponents,subintervals,shun', CASES,
                         ids=['left-root', 'left-pole', 'right-root',
                              'right-pole', 'both-poles', 'near-endpoints'])
def test_original_restrict_spotchecks(operator, exponents, subintervals, shun,
                                      record_property):
    # Native supplied exponents are used unchanged. singType is consulted
    # only for missing/NaN exponents in the native constructor.
    f = Singfun.from_function(operator, exponents)
    pieces = f.restrict(subintervals)
    count = len(subintervals)-1
    if count == 1:
        pieces = [pieces]
    for index, (left, right) in enumerate(zip(subintervals[:-1], subintervals[1:])):
        if index == 0 and shun[0] == 1:
            x = jnp.linspace(left+1e-4, right, 100)
        elif index == count-1 and shun[1] == 1:
            x = jnp.linspace(left, right-1e-4, 100)
        else:
            x = jnp.linspace(left, right, 100)
        mapped = (2/(right-left))*(x-left)-1
        exact = operator(x)
        actual = pieces[index](mapped)
        absolute = jnp.abs(exact)
        absolute = jnp.where(absolute < 1, 1., absolute)
        error = float(jnp.max(jnp.abs(exact-actual)))
        tolerance = float(jnp.max(5e4*absolute*jnp.finfo(jnp.float64).eps))
        record_property(f'piece_{index}_error', error)
        record_property(f'piece_{index}_native_bound', tolerance)
        assert error < tolerance
