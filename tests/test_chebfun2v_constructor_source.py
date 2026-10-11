"""Native test_constructor.m ten predicates executed on each of two domains.

Provenance
----------
MATLAB source: tests/chebfun2v/test_constructor.m
Chebfun commit: 7574c77
Python keyword-domain syntax adapts native trailing-domain arguments.
"""

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(
    scope="module",
    params=[(-1.0, 1.0, -1.0, 1.0), (-2.0, 3.0, 0.0, 1.0)],
    ids=["default", "nondefault"],
)
def source_family(request):
    d = request.param
    f1 = lambda x, y: jnp.cos(x * y)  # noqa: E731 - literal MATLAB function handle
    f2 = lambda x, y: jnp.sin(x) + jnp.cos(y)  # noqa: E731 - literal MATLAB function handle
    f3 = lambda x, y: jnp.exp(x) - jnp.exp(-y) + x  # noqa: E731 - literal MATLAB function handle
    g1 = Chebfun2.from_function(f1, domain=d)
    g2 = Chebfun2.from_function(f2, domain=d)
    g3 = Chebfun2.from_function(f3, domain=d)
    two = [
        Chebfun2v.from_functions(*args, domain=d)
        for args in [(f1, f2), (f1, g2), (g1, f2), (g1, g2)]
    ]
    three = [
        Chebfun2v.from_functions(*args, domain=d)
        for args in [
            (f1, f2, f3),
            (f1, f2, g3),
            (f1, g2, f3),
            (g1, f2, f3),
            (g1, g2, f3),
            (g1, f2, g3),
            (f1, g2, g3),
            (g1, g2, g3),
        ]
    ]
    return d, (g1, g2, g3), two, three


@pytest.mark.parametrize("clause", range(1, 11))
def test_literal_constructor(source_family, clause):
    _, _, two, three = source_family
    h1, h2 = (two[0], two[clause]) if clause <= 3 else (three[0], three[clause - 3])
    tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
    assert (h1 - h2).norm() < tol


def test_existing_components_identity_supplement(source_family):
    _, g, two, three = source_family
    assert two[3].components[0] is g[0].approx
    assert two[3].components[1] is g[1].approx
    assert three[7].components[2] is g[2].approx


def test_inferred_domain_supplement(source_family):
    d, g, _, _ = source_family
    h = Chebfun2v.from_functions(g[0], g[1])
    assert h.domain == d
    assert h.components[0] is g[0].approx


def test_one_component_supplement(source_family):
    _, g, _, _ = source_family
    h = Chebfun2v.from_functions(g[0])
    assert h.n_components == 1
    assert h.components[0] is g[0].approx


def test_vectorize_existing_components_supplement(source_family):
    _, g, _, _ = source_family
    h = Chebfun2v.from_functions(g[0], g[1], vectorize=True)
    assert h.components[0] is g[0].approx
    assert h.components[1] is g[1].approx


@pytest.mark.parametrize("outer_components", [0, 1, 2])
def test_single_complex_coordinate_compose_supplement(outer_components):
    """Literal native compose.m19-24 real/imag coordinate interpretation."""
    f = Chebfun2v.from_functions(lambda x, y: x / 4 + 1j * y / 4)
    a = lambda x, y: x + 2 * y + x * y  # noqa: E731 - literal MATLAB function handle
    b = lambda x, y: 1 - x + y  # noqa: E731 - literal MATLAB function handle
    if outer_components == 0:
        outer = Chebfun2.from_function(a)
        expected = Chebfun2.from_function(lambda x, y: x / 4 + y / 2 + x * y / 16)
        err = (f.compose(outer) - expected).norm()
    else:
        outer = Chebfun2v.from_functions(*([a] if outer_components == 1 else [a, b]))
        exact = [lambda x, y: x / 4 + y / 2 + x * y / 16]
        if outer_components == 2:
            exact.append(lambda x, y: 1 - x / 4 + y / 4)
        expected = Chebfun2v.from_functions(*exact)
        result = f.compose(outer)
        assert result.n_components == outer_components
        err = (result - expected).norm()
    assert float(err) < 100 * ChebfunPref().cheb2Prefs.chebfun2eps


def test_single_real_scalar_compose_supplement():
    f = Chebfun2v.from_functions(lambda x, y: x / 4)
    result = f.compose(jnp.sin)
    exact = Chebfun2.from_function(lambda x, y: jnp.sin(x / 4))
    assert float((result - exact).norm()) < 100 * ChebfunPref().cheb2Prefs.chebfun2eps


def test_single_real_cartesian_compose_supplement():
    """Native scalar-coordinate branch also maps real f to (f,0)."""
    f = Chebfun2v.from_functions(lambda x, y: x / 4)
    outer = Chebfun2.from_function(lambda x, y: x + 2 * y + x * y)
    result = f.compose(outer)
    exact = Chebfun2.from_function(lambda x, y: x / 4)
    assert float((result - exact).norm()) < 100 * ChebfunPref().cheb2Prefs.chebfun2eps


def test_single_complex_callback_rejection_supplement():
    """Native special complex allowance is limited to Chebfun2/2v outers."""
    f = Chebfun2v.from_functions(lambda x, y: x / 4 + 1j * y / 4)
    with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN2V:COMPOSE:Complex"):
        f.compose(jnp.sin)
