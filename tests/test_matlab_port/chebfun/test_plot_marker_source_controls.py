"""Source controls for Chebfun representation-grid plot markers.

Artist data are compared with independent NumPy Chebyshev evaluation; tolerances
below are diagnostic roundoff bounds, not MATLAB source test tolerances.

Provenance
----------
MATLAB source: @chebfun/plot.m, @chebfun/plotData.m,
    @chebtech/plotData.m, @bndfun/plotData.m,
    @chebfun/parsePlotStyle.m, @chebfun/getValuesAtBreakpoints.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.plotting import matlab_plot
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = np.finfo(float).eps
ROUND = 256 * EPS


def _nodes(n: int, kind: int) -> np.ndarray:
    """Independent cosine definition of ascending Chebyshev points."""
    if n == 1:
        return np.array([0.0])
    if kind == 1:
        return np.cos(np.pi * (np.arange(n) + 0.5) / n)[::-1]
    return np.cos(np.pi * np.arange(n) / (n - 1))[::-1]


def _coeff_values(coeffs, t):
    """Evaluate each Chebyshev coefficient column with NumPy's Clenshaw."""
    c = np.asarray(coeffs)
    if c.ndim == 1:
        return np.polynomial.chebyshev.chebval(t, c)
    return np.stack([
        np.polynomial.chebyshev.chebval(t, c[:, k])
        for k in range(c.shape[1])
    ], axis=-1)


def _physical_points(t, interval):
    a, b = interval
    return a + (np.asarray(t) + 1.0) * (b - a) / 2.0


def _make_fun(pieces, kind=2, *, point_values=None):
    """Build pieces from literal low-degree Chebyshev coefficients."""
    tech_type = Chebtech1 if kind == 1 else Chebtech2
    funs = [
        _Piece(tech=tech_type.from_coeffs(np.asarray(coeffs)),
               interval=interval)
        for coeffs, interval in pieces
    ]
    breaks = [pieces[0][1][0]] + [piece[1][1] for piece in pieces]
    f = Chebfun(funs=funs, domain=Domain(tuple(breaks)))
    return f if point_values is None else f.set_point_values(point_values)


def _marker_artists(ax):
    return [line for line in ax.lines
            if line.get_marker() not in (None, "None", "")]


def _line_artists(ax):
    return [line for line in ax.lines
            if line.get_marker() in (None, "None", "")]


def _finite_xy(artists):
    chunks = []
    for artist in artists:
        x = np.asarray(artist.get_xdata(), dtype=float).reshape(-1)
        y = np.asarray(artist.get_ydata()).reshape(-1)
        keep = np.isfinite(x) & np.isfinite(y)
        chunks.append(np.column_stack((x[keep], y[keep])))
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2))


def _assert_xy(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    # Sorting makes this independent of Matplotlib's artist creation order,
    # while retaining duplicate endpoint samples and array-column values.
    ai = np.lexsort((actual[:, 1], actual[:, 0]))
    ei = np.lexsort((expected[:, 1], expected[:, 0]))
    scale = max(1.0, float(np.max(np.abs(expected))) if expected.size else 1.0)
    np.testing.assert_allclose(actual[ai], expected[ei], rtol=0,
                               atol=ROUND * scale)


def _samples(coeffs, kind, interval, *, complex_plane=False):
    t = _nodes(len(np.asarray(coeffs)) if np.asarray(coeffs).ndim == 1
               else np.asarray(coeffs).shape[0], kind)
    values = np.asarray(_coeff_values(coeffs, t))
    if complex_plane:
        return np.column_stack((np.real(values).reshape(-1),
                                np.imag(values).reshape(-1)))
    x = _physical_points(t, interval)
    if values.ndim == 1:
        return np.column_stack((x, values))
    return np.concatenate([
        np.column_stack((x, values[:, k]))
        for k in range(values.shape[1])
    ], axis=0)


def _source_line_samples(coeffs, kind, interval, *, complex_plane=False):
    """Independent @chebtech/plotData degree grid for one smooth piece."""
    c = np.asarray(coeffs)
    n = c.shape[0]
    npts = min(max(501, int(np.floor(4.0 * np.pi * n + 0.5))), 65537)
    t = _nodes(npts, kind)
    values = np.asarray(_coeff_values(c, t))
    if complex_plane:
        return np.column_stack((np.real(values).reshape(-1),
                                np.imag(values).reshape(-1)))
    x = _physical_points(t, interval)
    if values.ndim == 1:
        return np.column_stack((x, values))
    return np.concatenate([
        np.column_stack((x, values[:, k]))
        for k in range(values.shape[1])
    ], axis=0)


def _assert_piece_separator(artists):
    """Source plotData concatenates each piece with a NaN separator."""
    assert any(
        np.any(~np.isfinite(np.asarray(a.get_xdata(), dtype=float)))
        for a in artists
    )


def _colors(artists):
    return {to_rgba(a.get_color()) for a in artists}


def _color_groups(artists):
    groups = {}
    for artist in artists:
        groups.setdefault(to_rgba(artist.get_color()), []).append(artist)
    return groups


def _match_marker_groups(markers, expected_groups):
    """Map each independent expected function/column dataset to one color."""
    groups = _color_groups(markers)
    remaining = set(groups)
    matches = []
    for expected in expected_groups:
        candidates = []
        for color in remaining:
            try:
                _assert_xy(_finite_xy(groups[color]), expected)
            except AssertionError:
                continue
            candidates.append(color)
        assert len(candidates) == 1
        matches.append(candidates[0])
        remaining.remove(candidates[0])
    assert not remaining
    return matches


def _assert_real_line_group_matches(artists, pieces):
    """Check line samples against the appropriate independent piece values."""
    checked = 0
    for artist in artists:
        if artist.get_linestyle() not in ("-", "solid"):
            continue
        x = np.asarray(artist.get_xdata(), dtype=float).reshape(-1)
        y = np.asarray(artist.get_ydata()).reshape(-1)
        for xx, yy in zip(x, y, strict=True):
            if not (np.isfinite(xx) and np.isfinite(yy)):
                continue
            possible = []
            for coeffs, (a, b) in pieces:
                if a - ROUND <= xx <= b + ROUND:
                    t = 2.0 * (xx - a) / (b - a) - 1.0
                    possible.append(float(_coeff_values(coeffs, np.array([t]))[0]))
            assert possible
            assert min(abs(float(yy) - value) for value in possible) <= ROUND * max(1.0, abs(float(yy)))
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("kind", [1, 2])
@pytest.mark.parametrize("fmt", ["o", "-o"])
def test_scalar_representation_markers_use_chebtech_nodes(kind, fmt):
    coeffs = np.array([0.2, -0.3, 0.15])
    f = _make_fun([(coeffs, (-2.0, 3.0))], kind)
    fig, ax = matlab_plot(f, fmt, numpts=31)
    try:
        markers = _marker_artists(ax)
        _assert_xy(_finite_xy(markers), _samples(coeffs, kind, (-2.0, 3.0)))
        assert all(a.get_linestyle() in ("None", " ") for a in markers)
        if fmt == "-o":
            line_only = _line_artists(ax)
            marker_color = _match_marker_groups(
                markers, [_samples(coeffs, kind, (-2.0, 3.0))]
            )[0]
            assert marker_color in _color_groups(line_only)
            _assert_real_line_group_matches(
                _color_groups(line_only)[marker_color],
                [(coeffs, (-2.0, 3.0))])
            assert any(a.get_linestyle() == "-" for a in _line_artists(ax))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fmt", ["o", "-o"])
def test_piecewise_marker_and_line_colors_are_reused_per_function(fmt):
    first = _make_fun([
        (np.array([0.1, 0.2]), (-1.0, 0.0)),
        (np.array([0.8, -0.15]), (0.0, 1.0)),
    ], kind=2)
    second_coeffs = np.array([-0.2, 0.1, 0.05])
    second = _make_fun([(second_coeffs, (-1.0, 1.0))], kind=2)
    fig, ax = matlab_plot([first, second], fmt, numpts=30)
    try:
        markers = _marker_artists(ax)
        lines = _line_artists(ax)
        expected = np.concatenate([
            _samples(np.array([0.1, 0.2]), 2, (-1.0, 0.0)),
            _samples(np.array([0.8, -0.15]), 2, (0.0, 1.0)),
            _samples(second_coeffs, 2, (-1.0, 1.0)),
        ])
        _assert_xy(_finite_xy(markers), expected)
        _assert_piece_separator(markers)
        expected_by_function = [
            np.concatenate([
                _samples(np.array([0.1, 0.2]), 2, (-1.0, 0.0)),
                _samples(np.array([0.8, -0.15]), 2, (0.0, 1.0)),
            ]),
            _samples(second_coeffs, 2, (-1.0, 1.0)),
        ]
        function_colors = _match_marker_groups(markers, expected_by_function)
        assert function_colors[0] != function_colors[1]
        if fmt == "-o":
            line_groups = _color_groups(lines)
            assert all(color in line_groups for color in function_colors)
            _assert_piece_separator(lines)
            _assert_real_line_group_matches(line_groups[function_colors[0]], [
                (np.array([0.1, 0.2]), (-1.0, 0.0)),
                (np.array([0.8, -0.15]), (0.0, 1.0)),
            ])
            _assert_real_line_group_matches(
                line_groups[function_colors[1]],
                [(second_coeffs, (-1.0, 1.0))])
    finally:
        plt.close(fig)


def test_explicit_breakpoint_override_does_not_replace_marker_samples():
    pieces = [
        (np.array([0.0, 0.25]), (-1.0, 0.0)),
        (np.array([1.0, -0.2]), (0.0, 1.0)),
    ]
    plain = _make_fun(pieces, kind=2)
    overridden = _make_fun(pieces, kind=2,
                           point_values=np.array([7.0, 19.0, -5.0]))
    fig0, ax0 = matlab_plot(plain, "-o", numpts=20)
    fig1, ax1 = matlab_plot(overridden, "-o", numpts=20)
    try:
        expected = np.concatenate([
            _samples(pieces[0][0], 2, pieces[0][1]),
            _samples(pieces[1][0], 2, pieces[1][1]),
        ])
        points0, points1 = _finite_xy(_marker_artists(ax0)), _finite_xy(_marker_artists(ax1))
        _assert_xy(points0, expected)
        _assert_xy(points1, expected)
        _assert_piece_separator(_marker_artists(ax1))
        marker_color = _match_marker_groups(_marker_artists(ax1), [expected])[0]
        line_groups = _color_groups(_line_artists(ax1))
        assert marker_color in line_groups
        _assert_piece_separator(_line_artists(ax1))
    finally:
        plt.close(fig0)
        plt.close(fig1)


def test_complex_piecewise_markers_plot_real_imaginary_representation_values():
    pieces = [
        (np.array([0.2 + 0.1j, 0.1 - 0.05j]), (-1.0, 0.0)),
        (np.array([-0.1 + 0.3j, 0.2 + 0.05j, -0.03j]), (0.0, 1.0)),
    ]
    f = _make_fun(pieces, kind=2,
                  point_values=np.array([3 + 4j, 8 - 2j, -5 + 6j]))
    fig, ax = matlab_plot(f, "-o", numpts=30)
    try:
        expected = np.concatenate([
            _samples(c, 2, interval, complex_plane=True)
            for c, interval in pieces
        ])
        markers = _marker_artists(ax)
        _assert_xy(_finite_xy(markers), expected)
        _assert_piece_separator(markers)
        line_expected = np.concatenate([
            _source_line_samples(c, 2, interval, complex_plane=True)
            for c, interval in pieces
        ])
        lines = _line_artists(ax)
        _assert_xy(_finite_xy(lines), line_expected)
        _assert_piece_separator(lines)
        marker_color = _match_marker_groups(markers, [expected])[0]
        line_groups = _color_groups(lines)
        assert marker_color in line_groups
        _assert_piece_separator(_line_artists(ax))
    finally:
        plt.close(fig)


@pytest.mark.parametrize("complex_values", [False, True])
def test_array_two_column_marker_series_preserve_column_data(complex_values):
    coeffs = np.array([
        [0.2, -0.1],
        [0.15, 0.2],
        [-0.04, 0.07],
    ])
    if complex_values:
        coeffs = coeffs.astype(complex)
        coeffs[:, 0] += 1j * np.array([-0.15, 0.08, 0.02])
        coeffs[:, 1] += 1j * np.array([0.3, -0.1, 0.05])
    f = _make_fun([(coeffs, (-1.5, 2.0))], kind=2)
    other_coeffs = np.array([-0.3, 0.07])
    if complex_values:
        other_coeffs = other_coeffs.astype(complex) + 1j * np.array([0.12, -0.04])
    other = _make_fun([(other_coeffs, (-1.5, 2.0))], kind=2)
    fig, ax = matlab_plot([f, other], "-o", numpts=24)
    try:
        expected_by_column = [
            _samples(coeffs[:, k], 2, (-1.5, 2.0),
                     complex_plane=complex_values)
            for k in range(2)
        ]
        expected_other = _samples(
            other_coeffs, 2, (-1.5, 2.0), complex_plane=complex_values)
        expected = np.concatenate([*expected_by_column, expected_other])
        markers = _marker_artists(ax)
        _assert_xy(_finite_xy(markers), expected)
        marker_colors = _match_marker_groups(
            markers, [*expected_by_column, expected_other])
        assert len(set(marker_colors)) == 3
        line_groups = _color_groups(_line_artists(ax))
        assert all(color in line_groups for color in marker_colors)
        if not complex_values:
            for k, color in enumerate(marker_colors[:2]):
                _assert_real_line_group_matches(
                    line_groups[color], [(coeffs[:, k], (-1.5, 2.0))])
            _assert_real_line_group_matches(
                line_groups[marker_colors[2]], [(other_coeffs, (-1.5, 2.0))])

        # Source plots each represented column as one series before moving to
        # the next input function. Compare with that scalar-column path so the
        # following function must receive the color after both columns.
        scalar_columns = [
            _make_fun([(coeffs[:, k], (-1.5, 2.0))], kind=2)
            for k in range(2)
        ]
        ref_fig, ref_ax = matlab_plot(
            [*scalar_columns, other], "-o", numpts=24)
        try:
            reference_colors = _match_marker_groups(
                _marker_artists(ref_ax),
                [*expected_by_column, expected_other])
            assert marker_colors == reference_colors
        finally:
            plt.close(ref_fig)
    finally:
        plt.close(fig)


def test_format_and_line_point_options_route_to_correct_artist_sets():
    coeffs = np.array([0.2, -0.3, 0.15])
    f = _make_fun([(coeffs, (-1.0, 1.0))], kind=2)
    fig, ax = matlab_plot(f, "-o", numpts=24, linewidth=3.0,
                          markersize=7.0, color="#663399")
    try:
        markers, lines = _marker_artists(ax), _line_artists(ax)
        assert markers and lines
        assert all(a.get_marker() == "o" for a in markers)
        assert all(a.get_markersize() == 7.0 for a in markers)
        assert all(a.get_linewidth() != 3.0 for a in markers)
        assert all(a.get_linewidth() == 3.0 for a in lines)
        assert _colors(markers) == {to_rgba("#663399")}
        assert _colors(lines) == {to_rgba("#663399")}
    finally:
        plt.close(fig)


def test_marker_keyword_without_format_is_point_only_and_supports_face_edge():
    f = _make_fun([(np.array([0.2, -0.3, 0.15]), (-1.0, 1.0))], kind=2)
    fig, ax = matlab_plot(f, numpts=24, marker="s", markersize=8.0,
                          markerfacecolor="red", markeredgecolor="green")
    try:
        markers, lines = _marker_artists(ax), _line_artists(ax)
        assert markers and lines
        assert all(a.get_marker() == "s" for a in markers)
        assert all(a.get_markersize() == 8.0 for a in markers)
        assert all(to_rgba(a.get_markerfacecolor()) == to_rgba("red")
                   for a in markers)
        assert all(to_rgba(a.get_markeredgecolor()) == to_rgba("green")
                   for a in markers)
        assert all(a.get_marker() in (None, "None", "") for a in lines)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fmt", ["red", "C2"])
def test_color_format_with_marker_keyword_keeps_default_line_and_marker_colors(fmt):
    f = _make_fun([(np.array([0.2, -0.3, 0.15]), (-1.0, 1.0))], kind=2)
    expected_color = to_rgba(fmt)
    fig, ax = matlab_plot(f, fmt, marker="d", lw=2.5, ms=6.0,
                          mfc="yellow", mec="black", numpts=24)
    try:
        markers, lines = _marker_artists(ax), _line_artists(ax)
        assert markers and lines
        marker_color = _match_marker_groups(
            markers, [_samples(np.array([0.2, -0.3, 0.15]), 2, (-1.0, 1.0))]
        )[0]
        assert marker_color == expected_color
        assert _colors(lines) == {expected_color}
        assert all(a.get_marker() == "d" and a.get_markersize() == 6.0
                   for a in markers)
        assert all(to_rgba(a.get_markerfacecolor()) == to_rgba("yellow")
                   for a in markers)
        assert all(to_rgba(a.get_markeredgecolor()) == to_rgba("black")
                   for a in markers)
        assert all(a.get_linewidth() == 2.5 for a in lines)
        assert all(a.get_linestyle() == "-" for a in lines)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("fmt", ["o", "-o"])
def test_interval_restriction_disables_representation_markers(fmt):
    f = _make_fun([(np.array([0.2, -0.3, 0.15]), (-1.0, 1.0))], kind=2)
    fig, ax = matlab_plot(f, fmt, interval=(-0.5, 0.5), numpts=24)
    try:
        assert not _marker_artists(ax)
        if fmt == "-o":
            assert any(a.get_linestyle() == "-" for a in _line_artists(ax))
        else:
            assert not any(a.get_linestyle() not in ("None", " ")
                           for a in _line_artists(ax))
    finally:
        plt.close(fig)
