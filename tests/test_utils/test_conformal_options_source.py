"""Source-derived optional-branch controls, separate from native numerical gates."""
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.plotting import chebfun_style, save_chebfun_figure
from chebfunjax.utils._conformal_options import plot_conformal
from chebfunjax.utils.conformal import conformal
from chebfunjax.utils.quadrature import chebpts

OUT = None


@pytest.fixture(autouse=True)
def _render_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], 'OUT', tmp_path)


def circle():
    return chebfun(lambda t: jnp.exp(1j*jnp.pi*t), trig=True)


def page():
    path = Path(__file__).resolve().parents[2] / 'examples/complex/conformal_mapping.py'
    spec = importlib.util.spec_from_file_location('qualified_conformal_page', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    save_chebfun_figure(fig, path, size=(600,253), dpi=100)
    assert Image.open(path).size == (600,253)
    return path


def test_native_grid_geometry():
    chebfun_style()
    plt.figure(figsize=(6,2.53), dpi=100)
    ctr = .2+.1j
    fig, axes = plot_conformal(circle(), ctr, 1, lambda z: ctr+z,
                               jnp.array([2+1j]), jnp.array([2j, -2j]))
    circ = np.exp(2j*np.pi*np.arange(301)/300)
    ray = np.asarray(chebpts(301))
    ray=ray[ray>=0]
    grids = [k/8*circ for k in range(1,8)]
    grids += [ray*np.exp(2j*np.pi*k/16) for k in range(1,17)]
    for index, ax in enumerate(axes):
        assert len(ax.lines) == 25
        assert ax.lines[0].get_linewidth() == 1
        assert ax.lines[1].get_markersize() == 8
        assert ax.lines[1].get_markeredgewidth() == 0
        marker = ax.lines[1]._marker
        assert marker.get_path().transformed(marker.get_transform()).get_extents().width == 1/3
        assert ax.get_title() == f'{index+1} poles'
        for artist, expected in zip(ax.lines[2:], grids, strict=True):
            values = artist.get_xdata()+1j*artist.get_ydata()
            np.testing.assert_allclose(values, expected+(ctr if index==0 else 0), rtol=0, atol=2e-15)
            assert artist.get_linewidth() == .5
        assert ax.title.get_fontweight() == 'normal'
    np.testing.assert_array_equal(axes[1].get_xticks(), [-1,0,1])
    np.testing.assert_allclose(axes[0].get_xlim(), [-1.2,1.6], rtol=0, atol=1e-15)
    np.testing.assert_allclose(axes[0].get_ylim(), [-1.3,1.5], rtol=0, atol=1e-15)
    save(fig, 'analytic_grid.png')
    bounds = [list(ax.get_window_extent().bounds) for ax in axes]
    np.testing.assert_allclose(bounds[0], [91.955,96.14,134.09,134.09], atol=1e-8)
    np.testing.assert_allclose(bounds[1], [379.955,96.14,134.09,134.09], atol=1e-8)
    (OUT/'geometry.json').write_text(json.dumps(bounds))
    plt.close(fig)


@pytest.mark.parametrize('method', ['ks','poly'])
def test_public_options_preserve_maps(method, capsys):
    C=circle()
    before=conformal(C, method='kerzman-stein' if method=='ks' else 'poly')
    assert capsys.readouterr().out == ''
    plt.figure(figsize=(6,2.53), dpi=100)
    after=conformal(C, method='kerzman-stein' if method=='ks' else 'poly', plots=method=='ks', numbers=True)
    output=capsys.readouterr().out
    z=jnp.array([0, .1+.2j, -.3j, .7])
    for a,b in zip(before[:2],after[:2],strict=True):
        np.testing.assert_array_equal(a(z),b(z))
        np.testing.assert_allclose(b(z),z,rtol=0,atol=1e-11)
    for a,b in zip(before[2:],after[2:],strict=True):
        np.testing.assert_array_equal(a,b)
    assert 'computation time in seconds:' in output
    assert ('plotting time in seconds:' in output) == (method=='ks')
    assert 'number of sample points Z on boundary, M:  '+('600' if method=='ks' else '128') in output
    assert 'interior inverse error norm(.9*W-f(finv(.9*W)),inf):' in output
    if method=='poly':
        assert 'polynomial degree, n:  16' in output
        assert 'number of real degrees of freedom, N:  33' in output
        assert 'condition number of least-squares matrix A:  1.4e+00' in output
    else:
        assert 'rough error measure, err:' in output
    OUT.mkdir(exist_ok=True)
    (OUT/f'numbers_{method}.txt').write_text(output)
    plt.close('all')


def test_page_second_geometry():
    mod=page()
    fig,ax=mod._plot_images(circle(),jnp.array([0,.2+.1j]))
    np.testing.assert_allclose(ax.get_position().bounds,[.13,.11,.775,.815],atol=1e-12)
    np.testing.assert_allclose(ax.get_ylim(),[-1.3,1.3],atol=1e-12)
    bbox=ax.get_window_extent()
    assert abs(bbox.width/np.diff(ax.get_xlim())[0]-bbox.height/2.6)<1e-10
    assert ax.lines[1].get_markersize()==3
    assert ax.lines[1].get_markeredgewidth()==0
    marker=ax.lines[1]._marker
    assert marker.get_path().transformed(marker.get_transform()).get_extents().width == 1/3
    # Export adds its documented1e-6pixel rounding allowance. Test exact
    # source aspect before that export mutation; verify actual PNG separately.
    save(fig,'analytic_scatter.png')
    np.testing.assert_allclose(ax.get_ylim(),[-1.3,1.3],atol=1e-12)
    plt.close(fig)


def test_explicit_sequential_jax_stream():
    mod=page()
    key,_=jax.random.split(jax.random.PRNGKey(0))
    key,a=jax.random.split(key)
    key,b=jax.random.split(key)
    expected=2*jax.random.uniform(a,(20000,),dtype=jnp.float64)-1+2j*jax.random.uniform(b,(20000,),dtype=jnp.float64)-1j
    expected=expected[jnp.abs(expected)<1][:10000]
    start,_=jax.random.split(jax.random.PRNGKey(0))
    actual=mod._disk_points(start)
    np.testing.assert_array_equal(actual,expected)
    assert actual.size==10000
