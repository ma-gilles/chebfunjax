"""Missing public REALLOG error branches, Chebfun 7574c77.

Source: @chebfun/reallog.m and @chebfun/compose.m; numeric negative-real
error identified from installed MATLAB R2025b elfun/reallog.m. These are
source API controls, not additional clauses of native test_log.m (positive5).
"""

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("case", ["constant", "interior", "array", "stored_point"])
@pytest.mark.parametrize("explicit_pref", [False, True])
def test_negative_real_error(case, explicit_pref):
    if case == "constant":
        f = cj.chebfun(lambda x: -jnp.ones_like(x))
    elif case == "interior":
        # Positive endpoints cannot substitute for the numeric callback gate.
        f = cj.chebfun(lambda x: x**2 - 0.25)
    elif case == "array":
        f = cj.chebfun(lambda x: jnp.stack((2 * jnp.ones_like(x), -jnp.ones_like(x)), axis=-1))
    else:
        # columnCompose applies the operator to stored pointValues separately.
        f = cj.chebfun(lambda x: 2 * jnp.ones_like(x)).define_point(0.0, -1.0)
    with pytest.raises(ValueError, match="MATLAB:reallog:complexResult"):
        f.reallog(ChebfunPref()) if explicit_pref else f.reallog()


@pytest.mark.parametrize("explicit_pref", [False, True])
def test_complex_input_error_precedes_composition(explicit_pref):
    f = cj.chebfun(lambda x: -2 + 1j * x)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN:reallog:complex"):
        f.reallog(ChebfunPref()) if explicit_pref else f.reallog()
