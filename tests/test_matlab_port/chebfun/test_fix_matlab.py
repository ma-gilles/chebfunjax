"""Seven literal native fix predicates; MATLAB RNG stream identity unverified.
Native tests/chebfun/test_fix.m, commit7574c77. Clause7 evaluates raw f,
not fix(f); retain that distinction and original strict threshold.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebpref import ChebfunPref

EPS = np.finfo(float).eps
DOM = (-1.0, -0.5, 0.5, 1.0)
_rng = np.random.RandomState(6178)
X = jnp.asarray(2 * _rng.rand(100) - 1)
X_UNBOUNDED = jnp.asarray((-3 * np.pi + 1e6) * _rng.rand(100) - 1e6)


def trunc_complex(values):
    return (
        jnp.trunc(jnp.real(values)) + 1j * jnp.trunc(jnp.imag(values))
        if jnp.iscomplexobj(values)
        else jnp.trunc(values)
    )


class TestChebfunFix:
    def test_native_empty(self):
        assert cj.chebfun().fix().isempty()

    @pytest.mark.parametrize("clause", [2, 3, 4, 5, 6])
    def test_native_ordinary_complex_array(self, clause):
        if clause == 2:
            op = lambda x: 2 * jnp.sin(x)
        elif clause == 3:
            op = jnp.exp
        elif clause == 4:
            op = lambda x: 1j * jnp.exp(x)
        elif clause == 5:
            op = lambda x: (1 + 1j) * jnp.exp(x)
        else:
            op = lambda x: jnp.stack([2 * jnp.sin(x), jnp.exp(x)], axis=-1)
        pref = ChebfunPref()
        if clause in (4, 5):
            f = cj.chebfun(jnp.exp, domain=DOM, pref=pref)
            f = 1j * f if clause == 4 else f + 1j * f
        else:
            f = cj.chebfun(op, domain=DOM, pref=pref)
        g = f.fix()
        error = float(jnp.max(jnp.abs(g(X) - trunc_complex(op(X)))))
        bound = 10 * g.vscale * EPS
        print({"source_clause": clause, "error": error, "source_bound": bound}, flush=True)
        assert error <= bound

    def test_native_raw_unbounded_clause7(self):
        f = cj.chebfun(lambda x: x * jnp.exp(x), domain=(-jnp.inf, -3 * jnp.pi))
        error = float(
            jnp.max(jnp.abs(f(X_UNBOUNDED) - jnp.trunc(X_UNBOUNDED * jnp.exp(X_UNBOUNDED))))
        )
        bound = EPS * f.vscale
        print({"source_clause": 7, "raw_function_error": error, "source_bound": bound}, flush=True)
        assert error < bound

    def test_fix_rounds_toward_zero(self):
        # Retained old port assertion: supplemental, not a native clause.
        f = cj.chebfun(lambda x: 2.5 * x, pref=ChebfunPref())
        xs = jnp.asarray([-0.9, -0.3, 0.3, 0.9])
        np.testing.assert_allclose(np.asarray(f.fix()(xs)), [-2.0, 0.0, 0.0, 2.0])


@pytest.mark.parametrize("transposed", [False, True])
def test_fix_native_integer_point_values_and_orientation(transposed):
    f = cj.chebfun(jnp.asarray([0.0, 2.0]), domain=(-2.0, 2.0), coeffs=True, pref=ChebfunPref())
    if transposed:
        f = f.transpose()
    g = f.fix()
    assert g.is_transposed == transposed
    np.testing.assert_array_equal(
        np.asarray(g(jnp.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]))),
        [-2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0],
    )
