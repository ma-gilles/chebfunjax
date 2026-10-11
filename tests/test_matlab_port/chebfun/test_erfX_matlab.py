"""Full native predicates from tests/chebfun/test_erfX.m, Chebfun 7574c77.

Five functions on scalar and three-column inputs use the original public
construction/subtraction routes, sample count, numeric infinity norms and
bounds. NumPy RandomState6178 is a stream adapter, not a fresh MATLAB RNG
identity claim. Six older sampled/error probes are retained as supplements.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf, erfc, erfinv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-0.97, 0.97, 100))


class TestChebfunErfX:
    @pytest.mark.parametrize(
        "name,meth,ref", [("erf", "erf", erf), ("erfc", "erfc", erfc), ("erfinv", "erfinv", erfinv)]
    )
    def test_erf_family_of_sin(self, name, meth, ref):
        f = cj.chebfun(jnp.sin)
        g = getattr(f, meth)()
        exact = jnp.asarray(ref(np.sin(np.asarray(X))))
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * max(g.vscale, 1.0), name

    def test_erfcx(self):
        # MATLAB test_erfX.m F{4} = @erfcx, composed with sin.
        from scipy.special import erfcx

        f = cj.chebfun(jnp.sin)
        g = f.erfcx()
        exact = jnp.asarray(erfcx(np.sin(np.asarray(X))))
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * max(g.vscale, 1.0)

    def test_erfcinv(self):
        # MATLAB test_erfX.m F{5} = @(x) erfcinv(x+1): the argument must
        # lie in [0, 2], so erfcinv is applied to sin + 1.
        from scipy.special import erfcinv

        f = cj.chebfun(jnp.sin) + 1.0
        g = f.erfcinv()
        exact = jnp.asarray(erfcinv(np.sin(np.asarray(X)) + 1.0))
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * max(g.vscale, 1.0)

    def test_erfcinv_complex_raises(self):
        # MATLAB erfcinv.m errors 'CHEBFUN:CHEBFUN:erfcinv:notreal'.
        f = cj.chebfun(lambda x: 1.0 + 1j * x)
        with pytest.raises(ValueError):
            f.erfcinv()


# Full original test_erfX.m predicates; previous probes above are supplemental.
import jax

from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("method", ["erf", "erfinv", "erfc", "erfcx", "erfcinv"])
@pytest.mark.parametrize("array_valued", [False, True])
def test_native_erfX_predicate(method, array_valued):
    pref = ChebfunPref()
    if method == "erfcinv":
        # Literal F{5}=@(x) erfcinv(x+1); existing JAX numeric kernel.
        numeric = lambda value: jax.scipy.special.erfinv(1 - (value + 1))
    else:
        numeric = getattr(jax.scipy.special, method)
    if array_valued:
        op = lambda x: jnp.stack((jnp.sin(x), jnp.exp(x - 1.1), x / 2), axis=-1)
        exact_op = lambda x: jnp.stack(
            (numeric(jnp.sin(x)), numeric(jnp.exp(x - 1.1)), numeric(x / 2)), axis=-1
        )
    else:
        op = jnp.sin
        exact_op = lambda x: numeric(jnp.sin(x))
    f = cj.chebfun(op, pref=pref)
    g = (f + 1).erfcinv() if method == "erfcinv" else getattr(f, method)()
    h = cj.chebfun(exact_op, pref=pref)
    # Source normest re-seeds each call and evaluates the constructed g-h.
    # Explicit NumPy stream adapter, no new MATLAB draw-equivalence claim.
    x = jnp.asarray(2 * np.random.RandomState(6178).rand(100) - 1)
    values = (g - h)(x)
    assert values.shape == ((100, 3) if array_valued else (100,))
    error = float(jnp.linalg.norm(values, ord=jnp.inf))
    bound = 10 * EPS * float(h.vscale) if array_valued else 10 * EPS
    print(
        {
            "method": method,
            "array": array_valued,
            "error": error,
            "bound": bound,
            "vscale_h": float(h.vscale),
            "shape": list(values.shape),
        },
        flush=True,
    )
    assert error < bound
