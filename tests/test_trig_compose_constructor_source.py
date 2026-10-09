"""Source compose-to-constructor ordering, Chebfun7574c77."""
import jax.numpy as jnp
import pytest

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = 2.**-52
PROBE = 2*.376989633393435-1


def _constant(n=4):
    return Trigtech.from_values(jnp.full(n, .25))


@pytest.mark.parametrize('binary', [False, True])
def test_nonstring_refiner_fixed_uses_raw_operator_grid(binary):
    f = _constant()
    calls = []

    def op(x):
        calls.append(x)
        return x

    def forbidden(*args):
        raise AssertionError('fixed construction must bypass custom refiner')

    result = f.compose(op, f if binary else None,
                       pref={'refinementFunction': forbidden, 'fixedLength': 5})
    expected = jnp.concatenate((trigpts(5), jnp.ones(1)))
    assert len(calls) == 1 and jnp.array_equal(calls[0], expected)
    assert result.n == 5
    assert jnp.array_equal(result.values, expected[:-1].at[0].set(0.))


@pytest.mark.parametrize('bad_probe', [False, True])
def test_nonstring_adaptive_raw_probe_precedes_refiner_error(bad_probe):
    calls = []

    def op(x):
        calls.append(x)
        return jnp.full_like(x, jnp.nan) if bad_probe else x

    with pytest.raises(ValueError, match='Cannot handle functions' if bad_probe else 'TRIGTECH:refine'):
        _constant().compose(op, pref={'refinementFunction': lambda *args: None})
    assert len(calls) == 1
    assert jnp.array_equal(calls[0], jnp.asarray([PROBE]))


def test_invalid_string_refiner_still_wraps_operand_before_probe():
    calls = []

    def op(x):
        calls.append(x)
        return x

    with pytest.raises(ValueError, match='TRIGTECH:refine'):
        _constant().compose(op, pref={'refinementFunction': 'not-a-refiner'})
    assert len(calls) == 1
    assert jnp.array_equal(calls[0], jnp.asarray([.25]))


@pytest.mark.parametrize('kind', ['strict', 'loose'])
def test_happiness_errors_follow_probe_and_first_grid(kind):
    sizes = []

    def op(x):
        sizes.append(x.size)
        return x

    with pytest.raises(ValueError, match=f'happinessCheck:{kind}Check'):
        _constant().compose(op, pref={'happinessCheck': kind})
    assert sizes == [1, 17]


def test_custom_checker_receives_source_overrides_and_private_fields():
    f = _constant(20)
    g = _constant(33)
    calls = []
    observed = []

    def op(x, y):
        calls.append(x.size)
        return x+y

    def checker(current, values, data, pref):
        assert current.n == 32 and current.ishappy is None
        assert pref['minSamples'] == 33
        assert pref['chebfuneps'] == EPS and pref['sampleTest'] is False
        assert data['hscale'] == 7 and data['tag']['x'] == 2
        assert values.shape == (32,)
        data['tag']['x'] = 9
        pref['maxLength'] = 0
        observed.append(True)
        return True, 1

    pref = {'minSamples': 2, 'chebfuneps': 1e-30, 'sampleTest': True,
            'happinessCheck': checker, 'maxLength': 33}
    data = {'hscale': 7, 'tag': {'x': 2}}
    result = f.compose(op, g, data=data, pref=pref)
    assert result.n == 1 and result.ishappy
    assert calls == [1, 33] and observed == [True]
    assert data == {'hscale': 7, 'tag': {'x': 2}}
    assert pref['minSamples'] == 2 and pref['chebfuneps'] == 1e-30
    assert pref['sampleTest'] is True and pref['maxLength'] == 33


@pytest.mark.parametrize('kind', ['classic', 'plateau'])
def test_classic_and_plateau_compose_actual_degree_one(kind):
    x = trigpts(16)
    f = Trigtech.from_values(jnp.cos(jnp.pi*x))
    result = f.compose(lambda y: y, pref={'happinessCheck': kind})
    assert result.ishappy and result.n == 3
    # Original source-scale trigonometric evaluation bound; not fitted.
    assert jnp.max(jnp.abs(result(x)-jnp.cos(jnp.pi*x))) < 10*EPS
