"""Literal ten optimization predicates, plus Python wrapper consistency.

Provenance
----------
MATLAB source: tests/chebfun3/test_optimization.m.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df.
All ten cases are qualified individually on CPU, including slow-marked cases.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


@pytest.mark.parametrize("case", [1,2,*[pytest.param(k,marks=pytest.mark.slow) for k in range(3,8)],8,9,10])
def test_original_optimization(case):
    pref=ChebfunPref()
    if case <= 7:
        op=lambda x,y,z:jnp.cos(case*jnp.pi*x*y*z)  # noqa: E731 — literal native anonymous function
    elif case == 8:
        op=lambda x,y,z:jnp.sin(jnp.pi*x*y*z)  # noqa: E731 — literal native anonymous function
    elif case == 9:
        op=lambda x,y,z:jnp.cos(0*jnp.pi*(x-y-z)**2)  # noqa: E731 — literal native anonymous function
    else:
        op=lambda x,y,z:jnp.sin(x+y+z)  # noqa: E731 — literal native anonymous function
    f=chebfun3(op,domain=(0,1,0,1,0,1))
    values,_=f.minandmax3()
    minimum=-1 if case<=7 else (1 if case==9 else 0)
    error=jnp.abs(values[0]-minimum)+jnp.abs(values[1]-1)
    assert float(error)<1000*pref.cheb3Prefs.chebfun3eps


def test_minandmax3_agrees_with_max3_and_min3():
    """Preserved supplemental Python wrapper check, outside native battery."""
    f = chebfun3(lambda x, y, z: jnp.cos(jnp.pi*x*y*z),
                 domain=(0., 1., 0., 1., 0., 1.))
    values, _ = f.minandmax3()
    tolerance = 1000*jnp.finfo(jnp.float64).eps
    assert abs(float(values[0])-float(f.min3()[0])) < tolerance
    assert abs(float(values[1])-float(f.max3()[0])) < tolerance
