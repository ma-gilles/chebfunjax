"""Native no-input empty representation and explicitly scoped Python adapters."""
import jax.numpy as jnp
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2


def test_omitted_versus_explicit_empty():
    f = Singfun.constructor()
    assert f.exponents == ()
    assert f.isempty()
    assert Singfun.constructor(f) is f
    numeric = Singfun.constructor(jnp.empty((0,)))
    smooth = Singfun.constructor(Chebtech2.empty())
    for other in (numeric, smooth):
        assert other.isempty()
        assert other.exponents == (0., 0.)
    assert Singfun.zeroSingFun().exponents == (0., 0.)
    assert not Singfun.zeroSingFun().isempty()


def test_empty_exponents_reject_nonempty_factor():
    with pytest.raises(ValueError, match='Empty exponents require'):
        Singfun(Chebtech2.from_values(jnp.array([1.])), ())


def test_native_empty_reductions_and_roots():
    f = Singfun.constructor()
    assert f.issmooth
    assert f.roots().shape == (0,)
    assert f.sum().shape == ()
    assert float(f.sum()) == 0.
    assert isinstance(f.cumsum(), jnp.ndarray)
    assert f.cumsum().shape == (0,)
    assert f.simplifyExponents() is f
    assert f.simplify().exponents == ()


@pytest.mark.parametrize('method', ['real', 'imag', 'conj', 'flipud', 'fliplr', 'diff'])
def test_native_empty_unary_preserves_exponents(method):
    out = getattr(Singfun.constructor(), method)()
    assert out.isempty()
    assert out.exponents == ()


@pytest.mark.parametrize("route", ["omitted", "numeric", "smooth"])
def test_native_unary_signs(route):
    f = (Singfun.constructor() if route == "omitted" else
         Singfun.constructor(jnp.empty((0,))) if route == "numeric" else
         Singfun.constructor(Chebtech2.empty()))
    for out in (+f, -f):
        assert out.isempty()
        assert out.exponents == f.exponents


@pytest.mark.parametrize('method', ['extractBoundaryRoots', 'cancelExponents'])
def test_python_empty_tech_adapters(method):
    # Native numeric[] dispatch for these methods is not qualified.
    f = Singfun.constructor()
    assert getattr(f, method)() is f
    assert 'exps=()' in repr(f)


def test_nonempty_method_tails():
    # Distinguishes accidental empty dispatch on a nonempty zero or smooth factor.
    f = Singfun(Chebtech2.from_values(jnp.array([2.])))
    assert f.exponents == (0., 0.)
    assert float(f.sum()) == 4.
    assert f.cumsum().n > 0
    assert not f.simplifyExponents().isempty()
    assert f.roots().shape == (0,)
