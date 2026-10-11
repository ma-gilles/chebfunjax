"""Coordinate API controls; no native test-count or MATLAB runtime claim."""

import numpy as np

from chebfunjax import cheb


def test_x():
    f = cheb.x()
    q = np.array([-1.0, -0.25, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(f(q), q, atol=32 * np.finfo(float).eps, rtol=0)
    assert tuple(f.domain.breakpoints) == (-1.0, 1.0)


def test_xy():
    fs = cheb.xy()
    assert len(fs) == 2
    a = np.array([-1.0, -0.25, 0.0, 0.5, 1.0])
    b = a[::-1]
    for f, expected in zip(fs, (a, b)):
        np.testing.assert_allclose(f(a, b), expected, atol=32 * np.finfo(float).eps, rtol=0)


def test_xyz():
    fs = cheb.xyz()
    assert len(fs) == 3
    a = np.array([-1.0, -0.25, 0.0, 0.5, 1.0])
    b = a[::-1]
    c = a / 2
    for f, expected in zip(fs, (a, b, c)):
        np.testing.assert_allclose(f(a, b, c), expected, atol=32 * np.finfo(float).eps, rtol=0)


def test_xydisk():
    fs = cheb.xydisk()
    assert len(fs) == 2
    t = np.array([-np.pi, -1.0, 0.0, 0.7, np.pi])
    r = np.array([1.0, 0.5, 0.0, 0.8, 1.0])
    for f, expected in zip(fs, (r * np.cos(t), r * np.sin(t))):
        np.testing.assert_allclose(f(t, r), expected, atol=32 * np.finfo(float).eps, rtol=0)


def test_xyzsphere():
    fs = cheb.xyzsphere()
    assert len(fs) == 3
    l = np.array([-np.pi, -1.0, 0.0, 0.7, np.pi])
    t = np.array([0.0, 0.5, np.pi / 2, 2.0, np.pi])
    for f, expected in zip(fs, (np.cos(l) * np.sin(t), np.sin(l) * np.sin(t), np.cos(t))):
        np.testing.assert_allclose(f(l, t), expected, atol=32 * np.finfo(float).eps, rtol=0)
