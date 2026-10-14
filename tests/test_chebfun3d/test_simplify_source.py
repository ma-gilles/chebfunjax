"""Supplemental native Chebfun7574c77 simplify controls; no original test exists."""
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.tech.trigtech import Trigtech

DATA = json.loads(Path(__file__).with_name('simplify_native.json').read_text())


def array(a):
    if isinstance(a, dict):
        return jnp.asarray(a['real']) + 1j*jnp.asarray(a['imag'])
    return jnp.asarray(a, dtype=jnp.float64)


def unpack(s):
    factors = []
    for key in ('cols', 'rows', 'tubes'):
        c = array(s[key])
        if not c.size:
            factors.append([])
            continue
        if c.ndim == 1:
            c = c[:, None]
        kind = {'chebtech1': Chebtech1, 'chebtech2': Chebtech2,
                'trigtech': Trigtech}[s.get('tech', 'chebtech2')]
        flags = s.get(key + '_real', [True]*c.shape[1])
        if not isinstance(flags, list):
            flags = [flags]
        factors.append([kind(coeffs=c[:, j], **(
            {'real_columns': (bool(flags[j]),)} if kind is Trigtech else {}))
            for j in range(c.shape[1])])
    rank = tuple(map(len, factors))
    return Chebfun3(*factors, array(s['core']).reshape(rank), tuple(s['domain']))


@pytest.mark.parametrize('case', [0, 1, 2, 'complex'])
def test_native_factor_panels(case):
    complex_case = case == 'complex'
    f = unpack(DATA['complex_input' if complex_case else 'factor_input'])
    ref = unpack(DATA['complex'] if complex_case else DATA['factor'][case])
    tol = 1e-6 if complex_case or case == 1 else None
    if case == 2:
        tol = jnp.empty((0,))
    result = f.simplify(tol)
    assert result.domain == f.domain
    assert jnp.array_equal(result.core, f.core)
    for actual, expected in zip((result.cols, result.rows, result.tubes),
                                (ref.cols, ref.rows, ref.tubes)):
        for a, b in zip(actual, expected):
            assert jnp.array_equal(a.coeffs, b.coeffs)


@pytest.mark.parametrize('case,tol', [(0, 0.), (1, .2), (3, 3.)])
def test_native_absolute_rank(case, tol):
    f = unpack(DATA['rank_input'])
    result = f.simplify(tol, 'rank')
    ref = unpack(DATA['rank'][case])
    assert result.domain == ref.domain
    assert result.core.shape == ref.core.shape
    assert result.isempty() == bool(DATA['rank'][case]['isempty'])
    if not result.isempty():
        x = jnp.asarray([0., .125, .5, 1.])
        assert jnp.max(jnp.abs(result(x, x, x)-ref(x, x, x))) < 200*jnp.finfo(jnp.float64).eps*2


@pytest.mark.parametrize('case', [0, 1, 2])
def test_native_empty_errors(case):
    assert not DATA['empty'][case]['ok']
    args = () if case == 0 else ((1e-6,) if case == 1 else (1e-6, 'rank'))
    with pytest.raises(ValueError if case == 2 else TypeError):
        Chebfun3.empty().simplify(*args)


def test_source_ignored_flag_and_extra_arguments():
    f = unpack(DATA['rank_input'])
    assert f.simplify(.2, 'arbitrary').core.shape == (1, 1, 1)
    assert f.simplify(.2, 'rank', 'extra') is f


def test_strict_threshold_uses_computed_singular_values():
    f = unpack(DATA['rank_input'])
    values, _ = f.hosvd()
    print('native_boundary_current_singular_values', [mode.tolist() for mode in values])
    result = f.simplify(2., 'rank')
    expected = tuple(int(jnp.sum(mode > 2.)) for mode in values)
    assert result.core.shape == expected
    # Native leading value is exactly 2; current JAX HOSVD rounds above it.
    # This source predicate is distinct from the retained failed native check.
    constant = Chebtech2(coeffs=jnp.asarray([1.]))
    exact = Chebfun3([constant], [constant], [constant],
                    jnp.asarray([[[2.]]]), (0., 1., 0., 1., 0., 1.))
    empty = exact.simplify(2., 'rank')
    assert empty.isempty()
    assert empty.core.shape == (0, 0, 0)
    assert empty.simplify() is empty


def test_native_exact_rank_threshold_two():
    test_native_absolute_rank(2, 2.)


@pytest.mark.parametrize('case', range(4))
@pytest.mark.parametrize('mode', [0, 1])
def test_native_other_tech_factor_panels(case, mode):
    data = json.loads(Path(__file__).with_name('simplify_native_tech.json').read_text())
    item = data[case]
    assert item['input']['tech'] == ('chebtech1' if case < 2 else 'trigtech')
    assert item['factor'][mode]['ok']
    f, ref = unpack(item['input']), unpack(item['factor'][mode])
    actual = f.simplify(None if mode == 0 else 1e-6)
    assert actual.domain == ref.domain
    assert jnp.array_equal(actual.core, ref.core)
    for aa, bb in zip((actual.cols, actual.rows, actual.tubes),
                      (ref.cols, ref.rows, ref.tubes)):
        for a, b in zip(aa, bb):
            assert type(a) is type(b)
            assert jnp.array_equal(a.coeffs, b.coeffs)
            if isinstance(a, Trigtech):
                assert a.real_columns == b.real_columns


@pytest.mark.parametrize('case', range(4))
def test_native_other_tech_rank(case):
    data = json.loads(Path(__file__).with_name('simplify_native_tech.json').read_text())
    item = data[case]
    assert item['input']['tech'] == ('chebtech1' if case < 2 else 'trigtech')
    assert item['rank']['ok']
    f, ref = unpack(item['input']), unpack(item['rank'])
    actual = f.simplify(1e-6, 'rank')
    assert actual.domain == ref.domain
    assert actual.core.shape == ref.core.shape
    t = jnp.linspace(0., 1., 11)
    xyz = [actual.domain[2*k] + t*(actual.domain[2*k+1]-actual.domain[2*k])
           for k in range(3)]
    expected = ref(*xyz)
    assert jnp.max(jnp.abs(actual(*xyz)-expected)) < 200*jnp.finfo(jnp.float64).eps*jnp.max(jnp.abs(expected))
    assert type(actual.cols[0]) is type(ref.cols[0])
