"""Original ASY-versus-REC assertion from Chebfun's Jacobi test.

Provenance
----------
MATLAB source: tests/misc/test_jacpts.m, clause 29.
Chebfun commit: 7574c77.
Other default ASY clauses 11-20 and 25 remain in test_jacpts_source.py.
"""

import jax.numpy as jnp

from chebfunjax.utils.quadrature import jacpts


def test_native_clause_29():
    x_asy, w_asy = jacpts(100, .1785, 2., [-1, 1], 'asy')
    x_rec, w_rec = jacpts(100, .1785, 2., [-1, 1], 'rec')
    bound = 100*jnp.finfo(jnp.float64).eps
    assert jnp.linalg.norm(x_asy-x_rec) < bound
    assert jnp.linalg.norm(w_asy-w_rec) < bound


def test_boundary_only_asy_at_source_dispatch_threshold():
    import json
    from pathlib import Path

    native = json.loads(Path(__file__).with_name('jacpts_n20_native.json').read_text())
    actual = jacpts(20, -.5, 0., 'asy', bary=True)
    rec = jacpts(20, -.5, 0., 'rec', bary=True)
    for value, name in zip((*actual, *rec), ('xa', 'wa', 'va', 'xr', 'wr', 'vr')):
        assert jnp.linalg.norm(value-jnp.asarray(native[name])) < 1e-12
    # The finite source ASY expansion itself misses this ideal comparison.
    # Preserve that observed behavior rather than substituting REC for ASY.
    assert jnp.linalg.norm(actual[0]-rec[0]) > 1e-12
    assert jnp.linalg.norm(jnp.asarray(native['xa'])-jnp.asarray(native['xr'])) > 1e-12


def test_explicit_default_and_asy_below_100():
    x, w, v = jacpts(42, -.1, .3, 'asy', bary=True)
    # Same original 42-point reference values and bounds; explicit ASY route.
    assert abs(w @ x-.363593965943934) < 1e-12
    assert abs(w @ (x*x)-.670376374709129) < 1e-12
    assert abs(x[36]-.912883347814032) < 1e-12
    assert abs(w[36]-.046661910947553) < 1e-12
    assert abs(v[36]-.320696510075909) < 1e-12
    explicit_default = jacpts(42, -.1, .3, [-1, 1], 'default', bary=True)
    for value, default_value in zip((x, w, v), explicit_default):
        assert jnp.array_equal(value, default_value)
