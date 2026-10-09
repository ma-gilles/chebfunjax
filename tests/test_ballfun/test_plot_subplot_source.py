"""Native @ballfun/plot.m changes current axes, not figure subplot spacing."""
import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np  # uses-numpy: exact Matplotlib host layout assertions.
import pytest

from chebfunjax.ballfun.ballfun import Ballfun
from chebfunjax.plotting import chebfun_style


def _ball():
    return Ballfun.from_coeffs(jnp.asarray([[[.25]]]), is_real=True)


def _layout(fig):
    return (
        tuple(getattr(fig.subplotpars, name)
              for name in ("left", "right", "bottom", "top", "wspace", "hspace")),
        tuple((tuple(ax.get_position(original=True).bounds),
               tuple(ax.get_position().bounds)) for ax in fig.axes),
    )


def _save(fig, name, tmp_path):
    output = Path(os.environ.get("CHEBFUN_RUNTIME_REPORT", tmp_path)) / "subplot_figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / name, dpi=100, bbox_inches=None)


@pytest.mark.parametrize("style,count", [("ball", 5), ("WedgeAz", 3), ("WedgePol", 5)])
def test_supplied_axes_and_siblings_keep_exact_positions(style, count, tmp_path):
    fig = plt.figure(figsize=(1.92, 1.92), dpi=100)
    try:
        ax = fig.add_subplot(221, projection="3d")
        fig.add_subplot(222)
        fig.add_subplot(223, projection="3d")
        fig.subplots_adjust(left=.17, right=.89, bottom=.16, top=.88,
                            wspace=.37, hspace=.31)
        fig.canvas.draw()
        before = _layout(fig)
        assert _ball().plot(style, ax=ax) == (fig, ax)
        assert len(ax.collections) == count
        assert ax.get_xlim() == ax.get_ylim() == ax.get_zlim() == (-1., 1.)
        aspect = ax.get_box_aspect()
        np.testing.assert_array_equal(aspect, np.full(3, aspect[0]))
        fig.canvas.draw()
        assert _layout(fig) == before
        _save(fig, f"supplied_{style}.png", tmp_path)
    finally:
        plt.close(fig)


def test_native_ten_subplot_indices_preserve_prior_allocations(tmp_path):
    fig = plt.figure(figsize=(6, 2.53), dpi=100)
    ball = _ball()
    try:
        params = _layout(fig)[0]
        previous = ()
        indices = []
        for l in range(4):
            for m in range(l + 1):
                index = l * 4 + m + 1
                ax = fig.add_subplot(4, 4, index, projection="3d")
                assert _layout(fig)[1][:-1] == previous
                before = _layout(fig)
                ball.plot(ax=ax)
                ax.set_axis_off()
                assert _layout(fig) == before
                previous = _layout(fig)[1]
                indices.append(ax.get_subplotspec().num1)
        assert indices == [0, 4, 5, 8, 9, 10, 12, 13, 14, 15]
        assert _layout(fig)[0] == params
        assert len(fig.axes) == 10
        assert sum(len(ax.collections) for ax in fig.axes) == 50
        fig.canvas.draw()
        assert _layout(fig)[1] == previous
        _save(fig, "ten_source_subplot_indices.png", tmp_path)
    finally:
        plt.close(fig)


def test_standalone_keeps_default_spacing_and_explicit_camera(tmp_path):
    with mpl.rc_context():
        chebfun_style()
        before = dict(mpl.rcParams)
        expected = tuple(mpl.rcParams[f"figure.subplot.{name}"]
                         for name in ("left", "right", "bottom", "top", "wspace", "hspace"))
        fig, ax = _ball().plot(elev=21, azim=-63)
        try:
            assert _layout(fig)[0] == expected
            assert (ax.elev, ax.azim) == (21, -63)
            assert len(ax.collections) == 5
            assert dict(mpl.rcParams) == before
            fig.canvas.draw()
            assert _layout(fig)[0] == expected
            _save(fig, "standalone.png", tmp_path)
        finally:
            plt.close(fig)
