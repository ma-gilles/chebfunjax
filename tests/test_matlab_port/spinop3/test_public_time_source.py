"""Public spin3 time/output controls from solvepde.m (Chebfun 7574c77).

Analytic field comparisons are supplemental; source checkpoint and matrix
shape semantics are literal. Native captured cases are qualified separately.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.spinop3 import Spinop3, spin3

DOM = (0., 2*np.pi, 0., 2*np.pi, 0., 2*np.pi)


def operator(times, system=False):
    s = Spinop3(DOM, times)
    if system:
        s.lin = '@(u,v) [.1*lap(u); .2*lap(v)]'
        s.nonlin = [lambda u, v: 0*u, lambda u, v: 0*v]
        s.init = [lambda x,y,z: jnp.cos(x), lambda x,y,z: jnp.sin(z)]
    else:
        s.lin = '@(u) .1*lap(u)'
        s.nonlin = lambda u: 0*u
        s.init = lambda x,y,z: jnp.cos(x)
    return s


@pytest.mark.parametrize('system', [False, True])
def test_three_requested_times(system):
    s = operator((0., .01, .02), system)
    solution, times = spin3(s, 8, .01, 'plot', 'off', return_times=True)
    np.testing.assert_array_equal(times, [0., .01, .02])
    assert solution.shape == (2 if system else 1, 3)
    for col, t in enumerate(times):
        assert abs(float(solution[0,col](.31,.27,.19))-np.cos(.31)*np.exp(-.1*float(t))) < 1e-12
        if system:
            assert abs(float(solution[1,col](.31,.27,.19))-np.sin(.19)*np.exp(-.2*float(t))) < 1e-12


def test_two_endpoints_return_times():
    solution, times = spin3(operator((0., .02)), 8, .01, return_times=True)
    np.testing.assert_array_equal(times, [0., .02])
    assert abs(float(solution(.31,.27,.19))-np.cos(.31)*np.exp(-.002)) < 1e-12


def test_off_step_request_is_not_interpolated():
    solution, times = spin3(operator((0., .015, .02)), 8, .01, return_times=True)
    np.testing.assert_array_equal(times, [0.])
    assert solution.shape == (1, 1)
    assert abs(float(solution[0,0](.31,.27,.19))-np.cos(.31)) < 1e-12


@pytest.mark.parametrize('times, expected_times', [
    ((0., .04, .05), [0., .04, .05]),
    ((0., .01, .05), [0.]),
])
def test_multistep_startup_checkpoint_order(times, expected_times):
    from chebfunjax.operators.spinpref import SpinPref3

    pref = SpinPref3(plot='off', M=16, scheme='abnorsett4')
    solution, actual_times = spin3(operator(times), 8, .01, pref, return_times=True)
    np.testing.assert_array_equal(actual_times, expected_times)
    assert solution.shape == (1, len(expected_times))
    assert pref.M == 16 and pref.scheme == 'abnorsett4'
    for col, t in enumerate(expected_times):
        assert abs(float(solution[0,col](.31,.27,.19))-np.cos(.31)*np.exp(-.1*t)) < 1e-12
