"""Preflush scale and readable geometry controls, not opaque renderer parity.

Provenance: @chebfun2v/quiver.m and quiver3.m, Chebfun
7574c77680d7e82b79626300bf255498271a72df; R2025b compatibility geometry.
Twelve expected-only records are bound to two verified source archives.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from chebfunjax._quiver_geometry import quiver_line_geometry
from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.plotting import _quiver_source_scale

FIX = Path(__file__).parents[2] / 'fixtures/quiver_preflush_scale_2025b'


def decode(item):
    assert item['isreal']
    values = [struct.unpack('>d', bytes.fromhex(word))[0]
              for word in item['real_hex'] if word]
    return jnp.reshape(jnp.asarray(values), item['size'], order='F')


@pytest.mark.parametrize('series', ['initial', 'disambiguating'])
@pytest.mark.parametrize('index', range(1, 7))
@pytest.mark.parametrize('disabled', [False, True])
def test_twelve_source_preflush_scale_factors(series, index, disabled):
    record = json.loads((FIX / f'{series}_{index}.json').read_text())
    inputs = [decode(record[key]) for key in ('XData', 'YData', 'UData', 'VData')]
    factor = decode(record['AutoScaleFactor']).reshape(())
    if not bool(decode(record['AutoScale']).reshape(())):
        factor = jnp.asarray(0.)
    kwargs = {}
    if 'ZData' in record and decode(record['ZData']).size:
        kwargs = {'z': decode(record['ZData']), 'w': decode(record['WData'])}
    expected = decode(record['ScaleFactor']).reshape(())
    with jax.disable_jit(disabled):
        actual = jax.jit(_quiver_source_scale)(*inputs, factor, **kwargs)
    assert bool(jnp.abs(actual - expected) <= jnp.spacing(expected))


@pytest.mark.parametrize('linewidth_kw', ['linewidth', 'linewidths', 'lw'])
def test_gradient_open_heads_axes_cycle_and_linewidth(linewidth_kw):
    f = chebfun2(lambda x, y: jnp.sin(2*x) + x*y*y)
    fig, ax = plt.subplots()
    try:
        ax.set_prop_cycle(color=['#123456', '#abcdef'])
        ax.plot([0, 1], [0, 1])
        assert f.gradient().quiver(ax=ax, n_pts=12, autoscale_factor=.5,
                                   **{linewidth_kw: .75}) == (fig, ax)
        shafts, heads = ax.collections[-2:]
        segments = np.asarray(shafts.get_segments())
        x, y = segments[:, 0, 0], segments[:, 0, 1]
        u, v = 2*np.cos(2*x)+y*y, 2*x*y
        scale = .5*np.sqrt(2*(2/12)**2)/np.max(np.hypot(u, v))
        np.testing.assert_allclose(segments[:, 1]-segments[:, 0],
                                   np.stack((u, v), axis=1)*scale, rtol=0, atol=1e-12)
        assert np.asarray(heads.get_segments()).shape == (144, 3, 2)
        np.testing.assert_allclose(shafts.get_colors()[0, :3],
                                   [0xab/255, 0xcd/255, 0xef/255], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(shafts.get_linewidths(), [.75])
        np.testing.assert_array_equal(ax.get_xlim(), [-1.1, 1.1])
        np.testing.assert_array_equal(ax.get_ylim(), [-1.1, 1.1])
    finally:
        plt.close(fig)


def test_default_factor_singleton_zero_and_native_opt_in():
    f = Chebfun2v.from_functions(lambda x, y: 1+0*x, lambda x, y: 2+0*y)
    fig, ax = plt.subplots()
    try:
        f.quiver(ax=ax, n_pts=1)
        np.testing.assert_allclose(ax.collections[-2].get_segments(),
                                   [[[1, 1], [1.9, 2.8]]], rtol=0, atol=1e-14)
        f.quiver(ax=ax, n_pts=1, autoscale_factor=0)
        np.testing.assert_allclose(ax.collections[-2].get_segments(),
                                   [[[1, 1], [2, 3]]], rtol=0, atol=1e-14)
        f.quiver(ax=ax, n_pts=3, scale=2, color='red')
        assert ax.collections[-1].scale == 2
        with pytest.raises(ValueError, match='conflicts'):
            f.quiver(ax=ax, autoscale_factor=.5, scale=2)
    finally:
        plt.close(fig)


def test_three_component_indirect_and_direct_sampling_with_all_components():
    f = Chebfun2v.from_functions(lambda x, y: 1+0*x,
                                 lambda x, y: .5+0*x, lambda x, y: 2+0*y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    try:
        assert f.quiver(ax=ax, n_pts=5, autoscale_factor=.5) == (fig, ax)
        segments = np.asarray(ax.collections[-2]._segments3d)
        assert segments.shape == (400, 2, 3)
        scale = .5*np.sqrt(2*(2/20)**2)/np.sqrt(1+.25+4)
        np.testing.assert_allclose(segments[:, 1]-segments[:, 0],
                                   np.tile(np.array([1, .5, 2])*scale, (400, 1)),
                                   rtol=0, atol=1e-14)
        assert f.quiver3(ax=ax, n_pts=5, autoscale_factor=.5) is ax
        assert len(ax.collections[-2]._segments3d) == 25
        f.quiver3(ax=ax, length=.25, n_pts=2)
        assert len(ax.collections[-1]._segments3d) == 3*4
    finally:
        plt.close(fig)


def test_direct_quiver3_two_component_dispatch_consumes_numpts():
    f = Chebfun2v.from_functions(lambda x, y: 1+0*x, lambda x, y: 2+0*y)
    fig, ax = plt.subplots()
    try:
        assert f.quiver3(ax=ax, n_pts=3) is ax
        assert len(ax.collections[-2].get_segments()) == 100
    finally:
        plt.close(fig)


@pytest.mark.parametrize('disabled', [False, True])
def test_readable_geometry_equality_order_and_zero(disabled):
    with jax.disable_jit(disabled):
        fn = jax.jit(quiver_line_geometry)
        shafts, heads, _ = fn(jnp.array([0.]), jnp.array([0.]),
                              jnp.array([2.]), jnp.array([0.]), 1.)
        np.testing.assert_array_equal(shafts, [[[0., 0.], [2., 0.]]])
        np.testing.assert_allclose(heads[0, (0, 2), 0], [1.67, 1.67],
                                   atol=4*np.finfo(float).eps, rtol=0)
        x = jnp.array([[0., 2.], [1., 3.]])
        shafts, heads, base = fn(x, jnp.zeros_like(x), jnp.ones_like(x), jnp.zeros_like(x))
        np.testing.assert_array_equal(base[:, 0], [0., 1., 2., 3.])
        assert heads.shape == (4, 3, 2)
        np.testing.assert_array_equal(heads[:, 1], shafts[:, 1])
        shafts, heads, _ = fn(jnp.array([0.]), jnp.array([0.]),
                              jnp.array([0.]), jnp.array([0.]))
        np.testing.assert_array_equal(shafts, np.zeros((1, 2, 2)))
        assert np.isnan(np.asarray(heads)[0, (0, 2)]).all()
        np.testing.assert_array_equal(heads[0, 1], [0., 0.])


def test_three_dimensional_compatibility_geometry():
    # Dyadic vector: cutoff=maxspan=2, norm=sqrt5.25>cutoff.
    shafts, heads, _ = quiver_line_geometry(
        jnp.array([0.]), jnp.array([0.]), jnp.array([1.]), jnp.array([.5]),
        1., z=jnp.array([0.]), w=jnp.array([2.]))
    # Here maxspan2 < norm=sqrt5.25; exact source norm2=norm/2.
    alpha = .33*2/np.sqrt(5.25)
    np.testing.assert_array_equal(shafts, [[[0., 0., 0.], [1., .5, 2.]]])
    np.testing.assert_allclose(heads[0, (0, 2), 2], [2-2*alpha]*2, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(heads[0, 1], [1., .5, 2.])


def test_source_preserves_aspect_and_explicit_colors_cycle():
    f = Chebfun2v.from_functions(lambda x, y: 1+0*x, lambda x, y: 2+0*y)
    fig, ax = plt.subplots()
    try:
        ax.set_prop_cycle(color=['red', 'blue'])
        ax.set_aspect('auto')
        f.quiver(ax=ax, n_pts=2, colors='green')
        assert ax.get_aspect() == 'auto'
        np.testing.assert_allclose(ax.collections[-2].get_colors()[0, :3],
                                   [0, 128/255, 0], atol=1e-15, rtol=0)
        f.quiver(ax=ax, n_pts=2)
        np.testing.assert_array_equal(ax.collections[-2].get_colors()[0, :3], [1, 0, 0])
        ax.set_aspect('equal')
        f.quiver(ax=ax, n_pts=2)
        assert ax.get_aspect() == 1
    finally:
        plt.close(fig)


def test_source_rejects_invalid_head_options():
    f = Chebfun2v.from_functions(lambda x, y: 1+0*x, lambda x, y: 2+0*y)
    fig, ax = plt.subplots()
    try:
        for size in [-1., float('nan'), float('inf'), 1j, [1, 2]]:
            with pytest.raises(ValueError, match='max_head_size'):
                f.quiver(ax=ax, n_pts=1, max_head_size=size)
        with pytest.raises(ValueError, match='show_arrow_head'):
            f.quiver(ax=ax, n_pts=1, show_arrow_head='off')
    finally:
        plt.close(fig)
