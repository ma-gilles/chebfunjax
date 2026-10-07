"""Independent factor-product metadata and following addition controls.

Provenance
----------
MATLAB source : @spherefun/times.m, @spherefun/plus.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.tech.trigtech import Trigtech


def _t(c):
    return Trigtech.from_coeffs(jnp.asarray(c), is_real=True)


@pytest.mark.parametrize('disable', [False, True])
@pytest.mark.parametrize('kind', ['constant', 'cosine', 'sine'])
def test_rank_one_product_preserves_source_metadata_and_addition(kind, disable, tmp_path):
    one = _t([1.0])
    cosine = _t([0.5, 0, 0.5])
    sine = _t([0.5j, 0, -0.5j])
    g = Spherefun(cols=[_t([3.0]), sine], rows=[one, cosine],
                  pivots=jnp.ones(2), idx_plus=(0,), idx_minus=(1,),
                  nonzero_poles=True, pivot_locations=((0.0, 0.0), (0.0, 1.0)))
    col = {'constant': _t([2.0]), 'cosine': cosine, 'sine': sine}[kind]
    minus = kind == 'sine'
    f = Spherefun(cols=[col], rows=[cosine if minus else one],
                  pivots=jnp.ones(1), idx_plus=() if minus else (0,),
                  idx_minus=(0,) if minus else (), nonzero_poles=not minus,
                  pivot_locations=((0.0, 0.0),))
    lam = jnp.asarray([-0.7, 0.2, 1.3])
    theta = jnp.asarray([0.0, 0.8, jnp.pi])
    fv = {'constant': 2*jnp.ones_like(theta), 'cosine': jnp.cos(theta),
          'sine': jnp.sin(theta)*jnp.cos(lam)}[kind]
    gv = 3+jnp.sin(theta)*jnp.cos(lam)
    with jax.disable_jit(disable):
        product = f*g
        summed = product+g
        product_values = product(lam, theta)
        sum_values = summed(lam, theta)
    output = Path(os.environ.get('HERMITE_OUTPUT', str(tmp_path))) / 'product_diagnostics'
    output.mkdir(exist_ok=True)
    record = {'kind': kind, 'disable_jit': disable,
              'lambda': np.asarray(lam).tolist(), 'theta': np.asarray(theta).tolist(),
              'expected_product': np.asarray(fv*gv).tolist(),
              'expected_sum': np.asarray((fv+1)*gv).tolist(), 'objects': {}}
    for name, obj in [('f', f), ('g', g), ('product', product), ('sum', summed)]:
        record['objects'][name] = {
            'pivots': np.asarray(obj.pivots).tolist(),
            'nonzero_poles': bool(obj.nonzero_poles),
            'idx_plus': list(obj.idx_plus), 'idx_minus': list(obj.idx_minus),
            'pivot_locations': list(obj.pivot_locations),
            'values': np.asarray(obj(lam, theta)).tolist(),
        }
    (output / f'{kind}_{disable}.json').write_text(json.dumps(record, indent=2)+'\n')
    # Source h=g: locations survive; flag is exactly the logical AND.
    assert product.nonzero_poles == (f.nonzero_poles and g.nonzero_poles)
    assert product.pivot_locations == g.pivot_locations
    assert product.idx_plus == (g.idx_minus if minus else g.idx_plus)
    assert product.idx_minus == (g.idx_plus if minus else g.idx_minus)
    np.testing.assert_allclose(product_values, fv*gv, rtol=0, atol=2e-12)
    np.testing.assert_allclose(sum_values, (fv+1)*gv, rtol=0, atol=2e-12)
