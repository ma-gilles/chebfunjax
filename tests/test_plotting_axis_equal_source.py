"""Literal MATLAB R2024b axis.m LocSetEqual mode transitions.

The rectangle fixtures exercise each unconstrained-axis branch independently.
No numerical Chebfun bounds are changed by these graphics controls.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chebfunjax.plotting import matlab_axis_equal


def _axes():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    return fig, ax


def test_equal_resets_horizontal_mode_and_centers_data():
    fig, ax = _axes()
    try:
        (line,) = ax.plot([0.8, 1.2], [-0.17, 0.17])
        ax.set_xlim(0, 1.5)
        ax.set_ylim(-0.5, 0.5)
        original = line.get_xydata().copy()
        assert matlab_axis_equal(ax) is ax
        fig.canvas.draw()
        np.testing.assert_allclose(ax.get_xlim(), [0, 2], rtol=0, atol=1e-14)
        assert ax.get_ylim() == (-0.5, 0.5)
        assert ax.get_autoscalex_on() and not ax.get_autoscaley_on()
        np.testing.assert_array_equal(line.get_xydata(), original)
    finally:
        plt.close(fig)


def test_equal_resets_vertical_mode_and_centers_data():
    fig, ax = _axes()
    try:
        ax.plot([-2, 2], [2, 4])
        ax.set_xlim(-6, 6)
        ax.set_ylim(-1, 1)
        matlab_axis_equal(ax)
        fig.canvas.draw()
        assert ax.get_xlim() == (-6, 6)
        np.testing.assert_allclose(ax.get_ylim(), [0, 6], rtol=0, atol=1e-14)
        assert not ax.get_autoscalex_on() and ax.get_autoscaley_on()
    finally:
        plt.close(fig)


def test_equal_ratio_takes_literal_vertical_else_branch():
    fig, ax = _axes()
    try:
        ax.plot([0, 2], [0, 1])
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 1)
        matlab_axis_equal(ax)
        fig.canvas.draw()
        assert ax.get_xlim() == (0, 2) and ax.get_ylim() == (0, 1)
        assert not ax.get_autoscalex_on() and ax.get_autoscaley_on()
        assert ax.get_position().bounds == (0.1, 0.1, 0.8, 0.8)
    finally:
        plt.close(fig)


def test_equal_empty_axes_retains_default_center():
    fig, ax = _axes()
    try:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        matlab_axis_equal(ax)
        fig.canvas.draw()
        assert ax.get_xlim() == (-0.5, 1.5) and ax.get_ylim() == (0, 1)
    finally:
        plt.close(fig)
