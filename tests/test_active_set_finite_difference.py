"""Native protected-provider captures and actual SQP first-iteration flows."""
import json
import math
import struct
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.utils._active_set_box_sqp import active_set_box
from chebfunjax.utils._active_set_finite_difference import finite_difference

FIXTURE = json.loads((Path(__file__).parent/'fixtures/active_set_fd_native.json').read_text())


def f64(value):
    return jnp.asarray(value, dtype=jnp.float64)


def words(value):
    return [value] if isinstance(value, str) else value


def decode(value):
    return [struct.unpack('>d', bytes.fromhex(v))[0] for v in words(value)]


def bits(value):
    return [struct.pack('>d', float(v)).hex() for v in jnp.ravel(value)]


def options():
    return {'FinDiffType': 'forward', 'FinDiffRelStep': jnp.full(2, jnp.sqrt(jnp.finfo(jnp.float64).eps)),
            'TypicalX': jnp.ones(2), 'DiffMinChange': 0., 'DiffMaxChange': float('inf'),
            'fwdFinDiff': True, 'scaleObjConstr': False, 'chkFunEval': False,
            'chkComplexObj': False, 'isGrad': True}


def assert_gradient(value, expected):
    # Native NaN payload/sign is not portable; finite and infinite words are exact.
    for actual, reference in zip(bits(value), words(expected), strict=True):
        if math.isnan(decode(reference)[0]):
            assert math.isnan(decode(actual)[0])
        else:
            assert actual == reference


@pytest.mark.parametrize('case', FIXTURE['direct'], ids=lambda c: c['name'])
def test_native_provider_transaction(case):
    x, lo, hi = (f64(decode(case[k])) for k in ('x_hex', 'lb_hex', 'ub_hex'))
    expected_inputs = words(case['inputs_hex'])
    expected_values = words(case['values_hex'])
    calls = []

    class NativeCallbackError(Exception):
        pass

    def replay_objective(point):
        index = len(calls)+1  # Native capture includes separately supplied base.
        calls.append(bits(point))
        assert calls[-1] == expected_inputs[2*index:2*index+2]
        if case['status'] == 'exception':
            raise NativeCallbackError(case['error_identifier'])
        return f64(decode(expected_values[index])[0])

    args = (x, replay_objective, lo, hi, f64(decode(case['base_hex'])[0]), options())
    if case['status'] == 'exception':
        with pytest.raises(NativeCallbackError, match=case['error_identifier']):
            finite_difference(*args)
        assert len(calls) == 1
    else:
        gradient, count = finite_difference(*args)
        assert count == len(calls) == case['returned_evaluations']
        assert_gradient(gradient, case['gradient_hex'])
    assert [v for call in calls for v in call] == expected_inputs[2:]


@pytest.mark.parametrize('case', FIXTURE['flow'], ids=lambda c: c['name'])
def test_native_first_sqp_flow_with_actual_objective(case):
    x, lo, hi = (f64(decode(case[k])) for k in ('x_hex', 'lb_hex', 'ub_hex'))
    points, values, gradients = [], [], []

    class NextIteration(Exception):
        pass

    def objective(point):
        value = point[0]**2 + 3*point[1]**2 + point[0]*point[1]
        points.extend(bits(point))
        values.extend(bits(value))
        return value

    def provider(*args):
        gradient, count = finite_difference(*args)
        gradients.append((bits(args[0]), gradient))
        return gradient, count

    def observer(event, iteration, state):
        # Native capture deliberately stops at MaxIter=1, after the next FD
        # transaction but before BFGS. This hook stops the unchanged engine
        # at the same callback boundary; no optimizer arithmetic is replaced.
        if event == 'bfgs' and iteration == 2:
            raise NextIteration

    try:
        result = active_set_box(objective, x, lo, hi, finite_difference=provider, observer=observer)
    except NextIteration:
        assert case['exitflag'] == 0
    else:
        assert result.exitflag == case['exitflag']
        assert result.iterations == case['iterations']
        assert result.evaluations == case['function_count']
        assert bits(result.x) == bits(f64(decode(case['point_hex'])))
        assert bits(result.value) == bits(f64(decode(case['value_hex'])[0]))
    assert points == words(case['inputs_hex'])
    assert values == words(case['values_hex'])
    assert len(values) == case['function_count']
    # Native public return can restore an earlier best-feasible point after
    # its MaxIter exit. Our pre-BFGS observer has not performed that return.
    # Match the returned point to its transaction, requiring an unambiguous
    # gradient when a point repeats; never choose a gradient by its value.
    returned_point = bits(f64(decode(case['point_hex'])))
    matches = [g for point, g in gradients if point == returned_point]
    assert matches
    assert all(bits(g) == bits(matches[0]) for g in matches)
    assert_gradient(matches[0], case['gradient_hex'])


@pytest.mark.parametrize('change', ['dimension', 'precision', 'nonfinite', 'width', 'options', 'step'])
def test_unsupported_provider_scope_rejected_before_callback(change):
    x, lo, hi, opts = f64([0., 0.]), f64([-1., -1.]), f64([1., 1.]), options()
    if change == 'dimension':
        x = f64([0.])
    elif change == 'precision':
        x = jnp.asarray([0., 0.], dtype=jnp.float32)
    elif change == 'nonfinite':
        x = f64([float('nan'), 0.])
    elif change == 'width':
        hi = lo
    elif change == 'options':
        opts['chkFunEval'] = True
    elif change == 'step':
        opts['FinDiffRelStep'] = f64([1e-6, 1e-6])
    calls = []
    with pytest.raises((TypeError, ValueError)):
        finite_difference(x, lambda point: calls.append(point), lo, hi, f64(0.), opts)
    assert calls == []
