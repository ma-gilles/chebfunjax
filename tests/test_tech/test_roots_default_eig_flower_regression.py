"""Untimed original flower inverse; retain prior postconstruction roundtrip gate."""
import importlib
import json
import os
from collections import Counter
from pathlib import Path

import jax.numpy as jnp

import chebfunjax as cj


def test_actual_flower_default_inverse(monkeypatch):
    tech = importlib.import_module('chebfunjax.tech.chebtech')
    inverse_module = importlib.import_module('chebfunjax.chebfun1d.inverse')
    original_leaf = tech._roots_default_eigenvalues
    original_brent = inverse_module._brent
    leaf_counts = []
    target_shapes = []

    def leaf(coeffs):
        leaf_counts.append(len(coeffs))
        return original_leaf(coeffs)

    def brent(f, y, *args, **kwargs):
        target_shapes.append(list(jnp.shape(y)))
        return original_brent(f, y, *args, **kwargs)

    monkeypatch.setattr(tech, '_roots_default_eigenvalues', leaf)
    monkeypatch.setattr(inverse_module, '_brent', brent)
    # Exact original flower construction from the qualified vscale diagnostic.
    t = cj.chebfun('t', domain=(0., 1.))
    flower = cj.exp(2j*jnp.pi*t) * (.5*cj.sin(8*jnp.pi*t)**2 + .5)
    forward = abs(flower.diff()).cumsum()
    result = forward.inv()
    # This is the prior postconstruction check, not an inverse target grid.
    x = jnp.linspace(0., 1., 129)
    error = float(jnp.max(jnp.abs(result(forward(x))-x)))
    output = os.environ.get('CHEBFUN_RUNTIME_REPORT')
    if output:
        (Path(output).parent/'flower_correctness.json').write_text(json.dumps({
            'forward_length': len(forward), 'inverse_length': len(result),
            'default_eig_input_counts': dict(Counter(leaf_counts)),
            'actual_brent_target_shapes': target_shapes,
            'roundtrip_error': error, 'unchanged_prior_bound': 1e-10,
            'domain': [float(v) for v in result.domain.breakpoints],
        }, indent=2)+'\n')
    assert leaf_counts
    assert error < 1e-10
