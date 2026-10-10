"""Original pinned Chebfun projectile inputs and bound, plus event observation."""
import json
import os
from pathlib import Path

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v


def test_native_projectile(monkeypatch, tmp_path):
    import chebfunjax.utils.native_ode45 as module
    original = module.native_ode45
    observed = []
    def capture(*args, **kwargs):
        sol = original(*args, **kwargs)
        observed.append(sol)
        return sol
    monkeypatch.setattr(module, 'native_ode45', capture)
    field = Chebfun2v.from_functions(lambda h, hp: hp, lambda h, hp: -1-.01*hp,
                     lambda h, hp: 1+0*h, domain=[0, 30, 0, 2])
    initial = jnp.asarray([2., 0., 0.])
    def p1Event1(t, y):
        return y[0]-y[2], 1, 0
    T, Y = field.ode45([0, 30], initial,
                      {'RelTol': 100*jnp.finfo(jnp.float64).eps, 'events': p1Event1})
    # Exact native pass(1), preserving Tend=30, options, and bound.
    assert abs(float(Y(0)[0])-float(initial[0])) < 1e-3
    assert len(observed) == 1
    sol = observed[0]
    report = {'span_requested': [0, 30], 'initial': [2, 0, 0],
              'last_time': float(sol['x'][-1]), 'event_times': sol['xe'].tolist(),
              'event_indices': sol['ie'].tolist(), 'stats': sol['stats'],
              'initial_height_error': abs(float(Y(0)[0])-2),
              'scope': 'Native initial-height predicate; event triggering separately observed, not assumed'}
    root = Path(os.environ.get('CHEBFUN_RUNTIME_REPORT', str(tmp_path)))
    root.mkdir(exist_ok=True)
    (root/'projectile.json').write_text(json.dumps(report, indent=2)+'\n')
