"""API and source branch controls for jacpts.m (Chebfun 7574c77)."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.utils.quadrature import jacpts


@pytest.mark.parametrize('method', ['REC', 'GW'])
def test_native_positional_method_forms(method):
    a = jacpts(25, .3, .5, method, bary=True)
    b = jacpts(25, .3, .5, [-1, 1], method.lower(), bary=True)
    for x, y in zip(a, b):
        assert jnp.array_equal(x, y)
    assert abs(jnp.sum(a[1])-1.652030414885670) < 1e-12


def test_explicit_rec_large_degree():
    x, w, v = jacpts(251, -.7, 1.3, 'REC', bary=True)
    assert abs(w@x-5.693053859952719) < 2e-12
    assert abs(w@(x*x)-5.218632704956658) < 2e-12
    assert abs(v[36]-.042040484981163) < 1e-12


def test_rec_jit_parameters():
    # Public host validation is intentionally not tracer-compatible; test kernel.
    from chebfunjax.utils.jacobi_points import _jacobi_rec
    x, w, v = jax.jit(lambda a, b: _jacobi_rec(42, a, b))(-.1, .3)
    assert bool(jnp.all(jnp.isfinite(x+w+v)))
    assert abs(x[36]-.912883347814032) < 1e-12


@pytest.mark.parametrize('method', ['ASY', 'GLR', 'invalid'])
def test_unsupported_methods_are_explicit(method):
    with pytest.raises((ValueError, NotImplementedError)):
        jacpts(42, -.1, .3, method)


def test_empty_and_single_bary():
    assert all(x.shape == (0,) for x in jacpts(0, .2, .3, bary=True))
    x, w, v = jacpts(1, .14, 1.1, [-2, 3], bary=True)
    assert x.shape == w.shape == v.shape == (1,)
    assert v[0] == 1


def test_source_small_gw_override():
    # Source dispatch selects REC for n<20 but skips C for explicit GW.
    from chebfunjax.utils.jacobi_points import _jacobi_rec
    x, w = jacpts(5, -.1, .3, 'GW')
    xr, wr, _ = _jacobi_rec(5, -.1, .3)
    assert jnp.array_equal(x, xr)
    assert jnp.array_equal(w, wr)


def test_mass_integer_table_and_noninteger_fallback():
    import json
    from pathlib import Path

    from jax.scipy.special import gammaln

    from chebfunjax.utils.jacobi_points import _jacobi_mass_gammaln

    table = jnp.asarray(json.loads(Path(__file__).with_name('gammaln_table.json').read_text()))
    actual = jax.jit(_jacobi_mass_gammaln)(jnp.arange(1., 101.))
    assert jnp.array_equal(actual, table)
    x = jnp.asarray([.14, .5, 1.1, 3.2, 100.1, 101.])
    assert jnp.array_equal(_jacobi_mass_gammaln(x), gammaln(x))
