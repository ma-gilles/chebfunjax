"""The thirty predicates of the pinned scalar Spherefun constructor test.

Provenance
----------
MATLAB source : tests/spherefun/test_constructor.m, @spherefun/constructor.m,
    @spherefun/sphf2cartf.m, @separableApprox/rank.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Original authors: Copyright 2017 by The University of Oxford
    and The Chebfun Developers.

Existing classmethod spellings adapt supported input forms. Unsupported source
constructor overloads require a public library factory; this adapter deliberately
fails rather than implementing the missing algorithms in the test. No skips,
xfails, sampled substitutes for continuous norms, or stored-rank substitutes.
"""

import inspect

import jax.numpy as jnp
import pytest

import chebfunjax.spherefun as sphere_api
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun import Spherefun
from chebfunjax.utils.quadrature import trigpts


def _construct(op, *source_options):
    """Syntax adapter; a future public factory owns unsupported semantics."""
    factory = getattr(sphere_api, "spherefun", None)
    if callable(factory):
        return factory(op, *source_options)
    if source_options == ("coeffs",):
        return Spherefun.coeffs2spherefun(op)
    if not source_options:
        if callable(op) and not isinstance(op, Spherefun):
            if len(inspect.signature(op).parameters) == 2:
                return Spherefun.from_function(op)
        elif not isinstance(op, (str, Spherefun)):
            return Spherefun.from_values(op)
    raise AssertionError(
        "Missing public spherefun source-input dispatch: Cartesian callback, "
        "vectorize, fixed rank/copy, eps option, or string input. Implement "
        "the library API; do not replace this call by a test-local algorithm."
    )


def _spherical(f):
    # Exactly the test's explicit redefine_function_handle/sphf2cartf(...,0).
    return lambda lam, th: f(jnp.cos(lam) * jnp.sin(th),
                            jnp.sin(lam) * jnp.sin(th), jnp.cos(th))


def _sample_error(f, g):
    x = trigpts(12, [-jnp.pi, jnp.pi])[0]
    y = jnp.linspace(0.0, jnp.pi, 6)
    lam, th = jnp.meshgrid(x, y)
    return jnp.max(jnp.abs(f(lam, th) - g.fevalm(x, y)))


def _tol():
    return 2e3 * ChebfunPref().cheb2Prefs.chebfun2eps


def _rank(f):
    method = getattr(f, "numerical_rank", None)
    assert method is not None, "Source rank requires the qualified numerical_rank API"
    return method()


def _error_id(call, expected):
    # Python ValueError text carries the MATLAB identifier; type is not a
    # substitute for the exact, case-sensitive identifier required by source.
    with pytest.raises(Exception) as caught:
        call()
    exc = caught.value
    identifier = getattr(exc, "identifier", str(exc).split()[0])
    assert identifier == expected


@pytest.mark.parametrize("clause,f,wrap_first", [
    (1, lambda x, y, z: x**2 + y**2 + z**2, True),
    (2, lambda x, y, z: jnp.exp(-jnp.cos(jnp.pi * (x + y + z))), True),
    (3, lambda x, y, z: 1 - jnp.exp(x), False),
    (4, lambda x, y, z: jnp.exp(y), False),
    (5, lambda x, y, z: jnp.exp(z), False),
    (6, lambda x, y, z: jnp.cos(x * y), False),
    (7, lambda x, y, z: jnp.sin(x * y * z), False),
    (8, lambda x, y, z: jnp.sin(x + y * z), True),
    (9, lambda x, y, z: jnp.sin(x + y * z) + 1, True),
], ids=[f"source_clause_{k:02d}" for k in range(1, 10)])
def test_constructor_sample_error(clause, f, wrap_first):
    spherical = _spherical(f)
    g = _construct(spherical if wrap_first else f)
    assert _sample_error(spherical, g) < _tol(), clause


def test_source_clause_10():
    g = _construct(lambda x, y, z: 0 * x)
    assert g.norm(jnp.inf) == 0


@pytest.mark.parametrize("clause,f", [
    (11, lambda x, y, z: jnp.cos(z)),
    (12, lambda x, y, z: x * y * z),
    (13, lambda x, y, z: 1),
], ids=["source_clause_11", "source_clause_12", "source_clause_13"])
def test_constructor_vectorize(clause, f):
    # Python * is elementwise; the public vectorize option must still be
    # accepted/executed. Scalar-only callback vectorization needs extra tests.
    a = _construct(f)
    b = _construct(f, "vectorize")
    assert (a - b).norm() < _tol(), clause


def test_source_clause_14():
    f = _construct(lambda x, y, z: 1 + x * jnp.sin(x * y))
    m, n = f.length()
    values = f.sample(m + m % 2, n)
    g = _construct(values)
    assert (f - g).norm() < _tol()


def test_source_clause_15():
    f = _construct(lambda x, y, z: 1 + 0 * x)
    g = _construct(jnp.ones((2, 2)))
    assert (f - g).norm() < _tol()


def test_source_clause_16():
    _error_id(lambda: _construct(jnp.ones((1, 2))),
              "CHEBFUN:SPHEREFUN:constructor:poleSamples")


def _gaussian(x, y, z):
    return jnp.exp(-10 * ((x - 1 / jnp.sqrt(2.0))**2
                         + (z - 1 / jnp.sqrt(2.0))**2 + y**2))


def test_source_clause_17():
    f = _construct(_gaussian)
    g = _construct(f.coeffs2(), "coeffs")
    assert (f - g).norm() < _tol()


@pytest.mark.parametrize("fixed_rank", [5, 6],
                         ids=["source_clause_18", "source_clause_19"])
def test_constructor_fixed_rank(fixed_rank):
    f = _construct(_gaussian, fixed_rank)
    assert _rank(f) == fixed_rank


def test_source_clause_20():
    f = _construct(_gaussian)
    g = _construct(f, 7)
    assert _rank(g) == 7


def test_source_clause_21():
    f = _construct(_gaussian)
    g = _construct(f, 0)
    assert _rank(g) == 0


def test_source_clause_22():
    f = _construct(_gaussian)
    g = _construct(f, 0)
    assert g.norm() < _tol()


def test_source_clause_23():
    _error_id(lambda: _construct(_gaussian, -1),
              "CHEBFUN:SPHEREFUN:constructor:parseInputs:domain3")


def test_source_clause_24():
    f = _construct(_gaussian)
    g = _construct(_gaussian, "eps", 1e-5)
    assert _rank(g) < _rank(f)


def test_source_clause_25():
    f = _construct(_gaussian)
    g = _construct(_gaussian, "eps", 1e-5)
    mf, nf = f.length()
    mg, ng = g.length()
    assert (mg < mf) and (ng < nf)


def test_source_clause_26():
    f = _construct(1)
    assert (f - 1).norm(jnp.inf) == 0


def test_source_clause_27():
    f = _construct(_gaussian)
    g = _construct("exp(-10*((x-1/sqrt(2)).^2 + (z-1/sqrt(2)).^2 + y.^2))")
    assert (f - g).norm(jnp.inf) == 0


def test_source_clause_28():
    f = _construct(lambda l, t: jnp.cos(l) * jnp.sin(t))
    g = _construct("cos(l).*sin(t)")
    assert (f - g).norm(jnp.inf) == 0


def test_source_clause_29():
    _error_id(lambda: _construct("x.*y.*z.*w"),
              "CHEBFUN:SPHEREFUN:constructor:str2op:depvars")


def test_source_clause_30():
    f = _construct(jnp.zeros((5, 4)))
    n, m = f.length()
    assert (m == 5) and (n == 4)
