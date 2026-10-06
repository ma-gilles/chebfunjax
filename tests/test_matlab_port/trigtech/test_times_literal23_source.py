"""All23 literal assertions from pinned MATLAB trigtech/test_times.m.

Provenance
----------
MATLAB source : tests/trigtech/test_times.m, @trigtech/isequal.m
Chebfun commit: 7574c77
Only seedRNG6178's100 query inputs come from a captured primitive fixture.
No source coefficients, output values or representation lengths are inputs.
The existing Python endpoint/accuracy regressions remain unchanged elsewhere.
"""
import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = np.finfo(np.float64).eps
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j


def _query():
    path = Path(__file__).resolve().parents[2] / 'fixtures' / 'trig_times_rng6178_2025b.json'
    fixture = json.loads(path.read_text())
    assert fixture['shape'] == [100, 1]
    words = np.asarray([int(word, 16) for word in fixture['query_words']], dtype=np.uint64)
    return jnp.asarray(words.view(np.float64))


def _norm_inf(a):
    # MATLAB norm(vector,inf) vs norm(matrix,inf): row-sum for matrices.
    a = jnp.asarray(a)
    return float(jnp.max(jnp.abs(a)) if a.ndim == 1 else
                 jnp.max(jnp.sum(jnp.abs(a), axis=1)))


def _scale(f):
    return float(jnp.max(f.vscale_columns()*EPS))


def _scalar_checks(f, op, alpha, x):
    a, b = f*alpha, alpha*f
    same = (a.coeffs.shape == b.coeffs.shape and
            bool(jnp.all(a.values == b.values)) and bool(jnp.all(a.coeffs == b.coeffs)))
    return same, _norm_inf(a(x)-op(x)*alpha) < 200*_scale(a)


def _product_checks(f, f_op, g, g_op, x, checkpos):
    h = f*g
    accuracy = _norm_inf(h(x)-f_op(x)*g_op(x)) < 1e5*_scale(h)
    if checkpos:
        # MATLAB relational >= uses the real part for complex storage.
        values = h.coeffs2vals(h.coeffs)
        return accuracy, bool(jnp.all(jnp.real(values) >= 0))
    return accuracy


def test_all_23_literal_matlab_assertions():
    x = _query()
    make = Trigtech.from_function  # Source factory defaults; no fixed n/maxpow override.
    f = Trigtech.empty()
    g = make(lambda x: jnp.sin(jnp.pi*x))
    assert (f*f).isempty() and (f*g).isempty() and (g*f).isempty(), 'pass1'

    f_op = lambda x: jnp.sin(jnp.cos(jnp.pi*x))
    f = make(f_op)
    result = _scalar_checks(f, f_op, ALPHA, x)
    assert result[0], 'pass2'
    assert result[1], 'pass3'
    f_op = lambda x: jnp.exp(jnp.stack((jnp.sin(jnp.pi*x), -jnp.cos(jnp.pi*x)), axis=-1))
    f = make(f_op)
    result = _scalar_checks(f, f_op, ALPHA, x)
    assert result[0], 'pass4'
    assert result[1], 'pass5'

    f_op = lambda x: 3/(4-jnp.cos(jnp.pi*x))
    f = make(f_op)
    g_op = lambda x: ALPHA*jnp.ones_like(x)
    g = make(g_op)
    assert _product_checks(f, f_op, g, g_op, x, False), 'pass6'
    f_op = lambda x: jnp.stack((jnp.sin(jnp.pi*x), jnp.cos(jnp.pi*x)), axis=-1)
    f = make(f_op)
    g_op = lambda x: jnp.tile(jnp.asarray([ALPHA, BETA]), (x.size, 1))
    g = make(g_op)
    assert _product_checks(f, f_op, g, g_op, x, False), 'pass7'

    f_op = lambda x: jnp.ones_like(x)
    f = make(f_op)
    assert _product_checks(f, f_op, f, f_op, x, False), 'pass8'
    f_op = lambda x: jnp.exp(jnp.cos(jnp.pi*x))-1
    f = make(f_op)
    g_op = lambda x: 3/(4-jnp.cos(jnp.pi*x))
    g = make(g_op)
    assert _product_checks(f, f_op, g, g_op, x, False), 'pass9'
    g_op = lambda x: jnp.cos(1e4*jnp.pi*x)
    g = make(g_op)
    assert _product_checks(f, f_op, g, g_op, x, False), 'pass10'
    g_op = lambda x: jnp.exp(1j*1e2*jnp.pi*x)
    g = make(g_op)
    assert _product_checks(f, f_op, g, g_op, x, False), 'pass11'

    f_op = lambda x: jnp.stack((jnp.sin(jnp.pi*x), jnp.cos(30*jnp.pi*x),
                               3/(4-jnp.cos(jnp.pi*x))), axis=-1)
    f = make(f_op)
    g_op = lambda x: jnp.tanh(jnp.sin(jnp.pi*x)+jnp.cos(jnp.pi*x))
    g = make(g_op)
    h1, h2 = f*g, g*f
    # Source uses norm(matrix) = spectral2; no prolong or max-entry substitute.
    assert float(jnp.linalg.norm(h1.coeffs-h2.coeffs, ord=2)) < 10*EPS, 'pass12'
    err = h1(x)-g_op(x)[:, None]*f_op(x)
    assert float(jnp.max(jnp.abs(err))) < 100*EPS, 'pass13'
    g_op = lambda x: jnp.stack((jnp.tanh(jnp.sin(jnp.pi*x)+jnp.cos(jnp.pi*x)),
                               jnp.sin(jnp.pi*x), jnp.exp(jnp.sin(jnp.pi*x))), axis=-1)
    g = make(g_op)
    h = f*g
    assert float(jnp.max(jnp.abs(h(x)-g_op(x)*f_op(x)))) < 100*EPS, 'pass14'
    g = make(lambda x: jnp.stack((jnp.sin(jnp.pi*x), jnp.cos(jnp.pi*x)), axis=-1))
    with pytest.raises(ValueError) as caught:
        _ = f*g
    # Python uses ValueError; retain the exact source identifier contract.
    identifier = getattr(caught.value, 'identifier', str(caught.value).split(': ', 1)[0])
    assert identifier == 'CHEBFUN:TRIGTECH:times:dim2', 'pass15'

    f_op = lambda x: jnp.exp(jnp.cos(jnp.pi*x))+jnp.exp(1j*2*jnp.pi*x)
    f = make(f_op)
    assert _product_checks(f, f_op, f, f_op, x, False), 'pass16'
    # Literal source closes over saved x despite naming the argument t.
    g_op = lambda t: jnp.conj(jnp.exp(jnp.cos(jnp.pi*x))+jnp.exp(1j*2*jnp.pi*x))
    g = f.conj()
    result = _product_checks(f, f_op, g, g_op, x, True)
    assert result[0], 'pass17'
    assert result[1], 'pass18'
    f_op = lambda x: 1+jnp.cos(jnp.pi*x)
    f = make(f_op)
    result = _product_checks(f, f_op, f, f_op, x, True)
    assert result[0], 'pass19'
    assert result[1], 'pass20'

    g_op = lambda x: 3/(4-jnp.cos(2*jnp.pi*x))
    g = make(g_op)
    h1 = f*g
    h2 = make(lambda x: f_op(x)*g_op(x))
    h2 = h2.prolong(h1.n)
    assert _norm_inf(h1.coeffs-h2.coeffs) < 50*EPS, 'pass21'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f = make(lambda x: jnp.exp(jnp.cos(jnp.pi*x)))
        g = make(lambda x: x)
        h = f*g
        assert not g.ishappy and not h.ishappy, 'pass22'
        h = g*f
        assert not g.ishappy and not h.ishappy, 'pass23'
