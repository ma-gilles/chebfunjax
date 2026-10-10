"""Complete private SQP returns and trajectories against original native files."""
import json
import os
import struct
from pathlib import Path

import jax.numpy as jnp
import pytest

from chebfunjax.utils._active_set_box_sqp import active_set_box
from chebfunjax.utils._active_set_finite_difference import finite_difference

FIXTURE = json.loads((Path(__file__).parent/'fixtures/active_set_full_native.json').read_text())


def f64(value):
    return jnp.asarray(value, dtype=jnp.float64)


def words(value):
    return [value] if isinstance(value, str) else value


def decode(value):
    return f64([struct.unpack('>d', bytes.fromhex(v))[0] for v in words(value)])


def bits(value):
    return [struct.pack('>d', float(v)).hex() for v in jnp.ravel(value)]


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda c: c['name'])
def test_complete_native_sqp_return_and_trajectory(case, tmp_path):
    name = case['name']
    x, lo, hi = (decode(case[k]) for k in ('x_hex', 'lower_hex', 'upper_hex'))
    points, values, providers, events = [], [], [], []
    capture = {'name': name}

    def objective(point):
        if name == 'interior':
            value = ((point[0]-.25)**2+(point[1]+.5)**2)/2
        elif name == 'corner':
            value = ((point[0]-2)**2+(point[1]+2)**2)/2
        elif name == 'equal_merit':
            value = (point[0]-1)**2+point[1]**2
        elif name == 'backtrack':
            value = 2*(point[0]-1)**2+point[1]**2
        elif name == 'negative_curvature':
            value = -point[0]**2
        elif name == 'stationary_restore':
            value = point[0]**2+3*point[1]**2+point[0]*point[1]
        elif name == 'infeasible_search':
            value = f64(jnp.any(point != 1))
            if bool(jnp.any((point < lo) | (point > hi))):
                value = f64(float(-2**100))
        else:
            raise AssertionError(name)
        points.extend(bits(point))
        values.extend(bits(value))
        return value

    def provider(*args):
        gradient, count = finite_difference(*args)
        providers.append({'point': bits(args[0]), 'gradient': bits(gradient),
                          'evaluations': len(values), 'extra_count': count})
        return gradient, count

    def observer(event, iteration, state):
        record = {'event': event, 'iteration': iteration}
        if event == 'bfgs':
            record.update(hessian=bits(state['hessian'].T),
                          corrected_y=bits(state['corrected_y']),
                          displacement=bits(state['displacement']))
        elif event == 'line_search':
            record.update(point=bits(state['x']), value=bits(state['value']),
                          trial=bits(state['trial']), evaluations=state['evaluations'])
        elif event == 'qp':
            qp = state['result']
            record.update(point=bits(qp.x), multipliers=bits(qp.multipliers),
                          active=list(qp.active), how=qp.how)
        events.append(record)

    try:
        result = active_set_box(objective, x, lo, hi, finite_difference=provider, observer=observer)
        capture.update(point=bits(result.x), value=bits(result.value), gradient=bits(result.gradient),
                       hessian=bits(result.hessian.T), multipliers=bits(result.multipliers),
                       optimality=bits(result.optimality), iterations=result.iterations,
                       evaluations=result.evaluations, exitflag=result.exitflag)
    finally:
        capture.update(inputs_hex=points, values_hex=values, providers=providers, events=events)
        directory = Path(os.environ.get('CHEBFUN_ACTIVE_SET_CAPTURE', str(tmp_path)))
        directory.mkdir(parents=True, exist_ok=True)
        (directory/(name+'.json')).write_text(json.dumps(capture, indent=2)+'\n')

    assert points == words(case['inputs_hex'])
    assert values == words(case['values_hex'])
    assert len(providers) == len(case['provider_events'])
    for actual, native in zip(providers, case['provider_events'], strict=True):
        assert actual['point'] == words(native['x_hex'])
        assert actual['gradient'] == words(native['gradient_hex'])
        assert actual['evaluations'] == native['values']['funccount']
        assert actual['extra_count'] == 2
    assert capture['point'] == words(case['x_final_hex'])
    assert capture['value'] == words(case['value_hex'])
    assert capture['gradient'] == words(case['gradient_hex'])
    assert capture['hessian'] == words(case['hessian_hex'])
    assert capture['multipliers'] == case['multipliers_hex']
    assert capture['optimality'] == case['optimality_hex']
    assert result.exitflag == case['exitflag']
    assert result.iterations == case['output']['iterations']
    assert result.evaluations == len(values) == case['actual_calls'] == case['output']['funcCount']
