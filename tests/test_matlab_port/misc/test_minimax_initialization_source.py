"""Literal source retry/state and rational clauses omitted by an aggregate xfail.

Provenance
----------
MATLAB source : minimax.m; tests/misc/test_minimax.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import contextlib
import importlib
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils import _minimax_init as initial
from chebfunjax.utils.minimax import _rational_kernel, minimax

mm = importlib.import_module('chebfunjax.utils.minimax')


def test_source_piecewise_linear_extrapolation_and_cdf_reference():
    x = jnp.asarray([-1., 0., 1.])
    y = jnp.asarray([10., 20., 50.])
    # Source histcounts maps either out-of-range side to its first interval.
    assert initial._piecewise_linear(x, y, jnp.asarray([-2., -.5, 1., 2.])).tolist() == [0., 15., 50., 40.]
    source = chebfun('x')
    actual = initial._reference_from_cdf(source, [-1., -.8, .2, 1.], 6, 0)
    np.testing.assert_allclose(actual, [-1., -.88, -.6, 0., .52, 1.], atol=3e-16, rtol=0)


@pytest.mark.parametrize('symmetry,reference,count,expected', [
    (1, [-1., -.4, .2, .7], 6, [-1., -.7, -.45, .2, .45, .7]),
    (1, [-.7, -.2, .4, 1.], 6, [-.7, -.45, -.2, .45, .7, 1.]),
    (2, [-1., -.5, 0., .3, .7], 7, [-1., -.7, -.5, -.3, .3, .5, .7]),
    (2, [-.7, -.3, 0., .5, 1.], 7, [-.7, -.3, 0., 0., .3, .7, 1.]),
])
def test_source_cdf_symmetry_keeps_literal_reflection_rules(symmetry, reference, count, expected):
    source = chebfun('x')
    actual = initial._reference_from_cdf(source, reference, count, symmetry)
    # These include the source's unpaired even node and repeated odd zero.
    np.testing.assert_allclose(actual, expected, atol=2e-16, rtol=0)


def test_source_dispatch_cf_aaa_cdf1_cdf2_sequence_and_authentic_messages(monkeypatch, capsys):
    source = chebfun('x')
    trace = []
    monkeypatch.setattr(initial, '_cf_reference', lambda *args: trace.append('cf') or [1])
    monkeypatch.setattr(mm, '_aaa_lawson_init', lambda *args: trace.append('aaa') or [2])
    monkeypatch.setattr(initial, '_cdf_reference', lambda *args: trace.append(f'cdf{args[-2]}') or [args[-2]+2])
    def kernel(m, n, reference, dialog=True):
        return SimpleNamespace(success=reference == [4])
    assert initial.initialize_rational(source, source, 2, 2, 0, kernel).success
    assert trace == ['cf', 'aaa', 'cdf1', 'cdf2']
    assert capsys.readouterr().out == 'Trying AAA-Lawson-based initialization...\n'


def test_source_dispatch_raises_after_all_trials_fail_and_silent_suppresses_text(monkeypatch, capsys):
    source = chebfun('x')
    monkeypatch.setattr(initial, '_cf_reference', lambda *args: [0])
    monkeypatch.setattr(mm, '_aaa_lawson_init', lambda *args: [0])
    monkeypatch.setattr(initial, '_cdf_reference', lambda *args: [0])
    with pytest.raises(RuntimeError, match='MINIMAX failed'):
        initial.initialize_rational(source, source, 2, 2, 0,
                                     lambda *a: SimpleNamespace(success=False), silent=True)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('fail_last', [False, True])
def test_source_kernel_retains_last_rational_trial_and_failure(monkeypatch, capsys, fail_last):
    calls = 0
    def trial(*args):
        nonlocal calls
        calls += 1
        if fail_last and calls == 2:
            return None, 1e-19, False, None, None, None
        return lambda x: jnp.zeros_like(jnp.asarray(x)), .1, True, np.asarray([-1., 1.]), np.asarray([1., 1.]), np.asarray([1., -1.])
    exchange_calls = 0
    def exchange(reference, *args):
        nonlocal exchange_calls
        exchange_calls += 1
        new = np.asarray(reference).copy()
        new[1] += .01
        return new, .2 if exchange_calls == 1 else .3, 1
    monkeypatch.setattr(mm, '_compute_trial_rational', trial)
    monkeypatch.setattr(mm, '_exchange_rat', exchange)
    monkeypatch.setattr(mm, '_pzeros', lambda *args: (np.asarray([]), np.asarray([])))
    reference = np.asarray([-1., -.3, .3, 1.])
    with pytest.warns(RuntimeWarning) if not fail_last else contextlib.nullcontext():
        result = _rational_kernel(lambda x: jnp.ones_like(x), 1, 1,
                                  domain=(-1., 1.), init_xk=reference, max_iter=2)
    assert result.success is (not fail_last)
    if fail_last:
        assert result.r is None
        assert np.isinf(result.err)
        assert result.delta == .1
        assert result.support.size == 0
        assert capsys.readouterr().out == 'Trial interpolant too far from optimal...\n'
    else:
        assert result.err == .3
        assert result.xk[1] == reference[1]+.01+.01


@pytest.mark.parametrize('clause', [3, 4, 7, 8, 9, 11, 13, 17, 18, 19])
def test_original_rational_clauses_at_source_bounds(clause):
    x = chebfun('x')
    # Preserve the established port seed; no native MATLAB random-stream claim.
    xx = jnp.asarray(2*np.random.RandomState(6178).rand(100)-1)
    if clause == 3:
        f = ((x+3)*(x-.5))/(x**2-4)
        r = minimax(f, 2, rational=True, denom=2, tol=1e-12, max_iter=20)
        assert float(jnp.max(jnp.abs(f(xx)-r.r(xx)))) < 1e-10
    elif clause == 4:
        x = chebfun('x', domain=(-1., 0., 1.))
        f = ((x-3)*(x+.2)*(x-.7))/((x-1.5)*(x+2.1))
        r = minimax(f, 3, rational=True, denom=2)
        assert float(jnp.max(jnp.abs(f(xx)-r.r(xx)))) < 1e-10
    elif clause in (7, 8):
        f = abs(x)
        result = minimax(f, 0 if clause == 7 else 2, rational=True, denom=0 if clause == 7 else 2)
        p, q = result.as_chebfuns()
        error = float((f-p/q).norm(jnp.inf))
        assert abs(error-(.5 if clause == 7 else .043689)) < (1e-10 if clause == 7 else 1e-3)
    elif clause == 9:
        minimax(x**3, 0, rational=True, denom=2)
    elif clause == 11:
        f1, f2 = chebfun('exp(x)'), chebfun('1e-100*exp(x)')
        r1 = minimax(f1, 1, rational=True, denom=3)
        r2 = minimax(f2, 1, rational=True, denom=3)
        sample1, sample2 = float(r1.r(.3)-f1(.3)), float(r2.r(.3)-f2(.3))
        assert abs(sample1-sample2*1e100) < 1e-3
    elif clause == 13:
        result = minimax(abs(x), 30, rational=True, denom=30, silent=True)
        assert abs(result.err-2.1739878e-7)/2.1739878e-7 < 1e-3
    elif clause == 17:
        f = abs(x-.1).sqrt()
        result = minimax(f, 4, rational=True, denom=4, silent=True)
        xx = jnp.linspace(-1, 1, 10000)
        error = float(jnp.max(jnp.abs(f(xx)-result.r(xx))))
        assert abs(result.err-error)/result.err < 1e-4
    elif clause == 18:
        result = minimax(1e40*abs(x), 5, rational=True, denom=5)
        assert result.err < 1e38
    else:
        result = minimax(jnp.sqrt, 4, rational=True, denom=4, domain=(0., 1.))
        p, q = result.as_chebfuns()
        zer1 = np.sort_complex(np.asarray(p.roots(all_roots=True)))
        pol1 = np.sort_complex(np.asarray(q.roots(all_roots=True)))
        zer2, pol2 = np.sort_complex(result.zeros), np.sort_complex(result.poles)
        assert np.max(np.abs(zer1-zer2))/np.max(np.abs(zer1)) < 1e-5
        assert np.max(np.abs(pol1-pol2))/np.max(np.abs(pol1)) < 1e-5


def test_source_odd_degree_default_iterations_round_half_away(monkeypatch):
    captured = {}
    monkeypatch.setattr(mm, '_adjust_degrees_for_symmetries', lambda *args: (5, 4, 0))
    def kernel(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(success=True)
    monkeypatch.setattr(mm, '_rational_kernel', kernel)
    monkeypatch.setattr(initial, 'initialize_rational',
                        lambda source, handle, m, n, symmetry, run, **kw: run(m, n, [0]))
    assert minimax(chebfun('exp(x)'), 5, rational=True, denom=4).success
    assert captured['max_iter'] == 13


def test_source_first_trial_failure_keeps_initial_delta(monkeypatch):
    monkeypatch.setattr(mm, '_compute_trial_rational',
                        lambda *args: (None, 0., False, None, None, None))
    result = _rational_kernel(lambda x: jnp.ones_like(x), 1, 1,
                              domain=(-1., 1.), init_xk=[-1., -.3, .3, 1.], silent=True)
    assert not result.success
    assert np.isinf(result.err)
    assert result.delta == 1.
    assert result.iter == 1
