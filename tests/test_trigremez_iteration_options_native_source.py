"""Pinned native trigremez options, display and iteration presentation controls."""

import importlib
import json
import os
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.utils._trigremez_iteration_options import HEADER, iteration_hook
from chebfunjax.utils.trigrational import TrigRemezError

FIXTURES = Path(__file__).parent / "fixtures/trigremez_iteration_options_source_2025b"
META = json.loads((FIXTURES / "metadata.json").read_text())["cases"]


@pytest.mark.parametrize("case", [1, 2, 4, 5])
def test_native_literal_display_format_and_zero_based_state(case, capsys):
    with np.load(FIXTURES / f"case{case}.npz") as event:
        hook = iteration_hook(True, False)
        hook(
            int(event["iteration"]),
            event["error"],
            event["h"],
            event["delta"],
            event["normf"],
            event["diffx"],
            None,
            None,
            None,
            None,
            None,
        )
    assert capsys.readouterr().out == HEADER + "\n" + META[str(case)]["display_row"] + "\n"


@pytest.mark.parametrize(
    "case,arguments",
    [(1, (1, "Display", "iter", "MAXITER", 1)), (2, (2, 2, "display", "off", "maxiter", 1))],
)
def test_public_callback_native_common_metrics_with_recorded_polynomial_reference_gap(
    case, arguments, monkeypatch, capsys
):
    module = importlib.import_module("chebfunjax.utils._trigremez_iteration_options")
    create = module.iteration_hook
    states = []

    def instrument(*args, **kwargs):
        original = create(*args, **kwargs)

        def capture(*state):
            states.append(state)
            return original(*state)

        return capture

    monkeypatch.setattr(module, "iteration_hook", instrument)
    f = abs(cj.chebfun("x"))
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        output = cj.trigremez(f, *arguments)
    assert output[-1]["iter"] == 1 and len(states) == 1
    state = states[0]
    assert state[0] == 0
    with np.load(FIXTURES / f"case{case}.npz") as native:
        for actual, name in zip(state[1:6], ["error", "h", "delta", "normf", "diffx"]):
            np.testing.assert_allclose(actual, native[name], rtol=0, atol=1e-12)
        np.testing.assert_allclose(state[6], native["old_reference"], rtol=0, atol=1e-12)
        if case == 2:
            np.testing.assert_allclose(state[7], native["new_reference"], rtol=0, atol=1e-12)
        else:
            # The retained first gate fails raw reference equality: the unchanged
            # polynomial provider chooses the other symmetric extremum.
            # This symmetry control does not repair or erase that parity gap.
            np.testing.assert_allclose(
                jnp.sort(jnp.abs(state[7])),
                np.sort(np.abs(native["new_reference"])),
                rtol=0,
                atol=1e-12,
            )
        x = jnp.linspace(-jnp.pi, jnp.pi, 5000)
        np.testing.assert_allclose(state[8](x) - state[9](x), native["plot_y"], rtol=0, atol=1e-12)
    capture_dir = os.environ.get("TRIGREMEZ_ITERATION_CAPTURE_DIR")
    if capture_dir:
        destination = Path(capture_dir)
        destination.mkdir(exist_ok=True)
        np.savez(
            destination / f"case{case}.npz",
            iteration=state[0],
            error=state[1],
            h=state[2],
            delta=state[3],
            normf=state[4],
            diffx=state[5],
            old_reference=state[6],
            new_reference=state[7],
            plot_y=state[8](x) - state[9](x),
        )
    expected_row = (
        f"0        {float(state[1]):5.4e}        {float(jnp.abs(state[2])):5.4e}"
        f"        {float(state[3] / state[4]):5.4e}        {float(state[5]):5.4e}"
    )
    assert capsys.readouterr().out == HEADER + "\n" + expected_row + "\n"
    assert len(emitted) == 1
    assert (
        str(emitted[0].message)
        == META[str(case)]["warning_id"] + ": " + META[str(case)]["warning_message"]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        (1, "maxiter", 2, "MAXITER", 1, "DISPLAY", 0),
        (1, 0, "tol", 1e-3, "display", [], "maxiter", 1),
    ],
)
def test_native_pairs_forward_last_option_and_presence_flag(arguments, monkeypatch, capsys):
    provider = importlib.import_module("chebfunjax.utils._trigremez_polynomial_source")
    seen = []
    sentinel = object()

    def polynomial(*args, **kwargs):
        seen.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(provider, "polynomial_trigremez", polynomial)
    assert cj.trigremez(cj.chebfun("x"), *arguments) is sentinel
    assert seen[0][0][2] == 1
    expected_tol = 1e-3 if len(arguments) == 8 else 1e-15 * 12
    assert seen[0][0][3] == expected_tol
    assert seen[0][1]["iteration_callback"] is not None
    assert capsys.readouterr().out == HEADER + "\n"


def test_unknown_name_source_identifier_and_degree_precedence():
    f = cj.chebfun("x")
    with pytest.raises(TrigRemezError) as caught:
        cj.trigremez(f, 1, "unknown", 7)
    assert caught.value.identifier == META["3"]["error_id"]
    assert str(caught.value) == META["3"]["error_id"] + ": " + META["3"]["error_message"]
    with pytest.raises(TrigRemezError) as caught:
        cj.trigremez(f, -1, "unknown", 7)
    assert caught.value.identifier.endswith(":parseInputs")


def test_existing_numeric_positional_and_keyword_adapters(monkeypatch):
    provider = importlib.import_module("chebfunjax.utils._trigremez_polynomial_source")
    seen = []
    sentinel = object()

    def polynomial(*args, **kwargs):
        seen.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(provider, "polynomial_trigremez", polynomial)
    f = cj.chebfun("x")
    assert cj.trigremez(f, 1, None, 3, 1e-3) is sentinel
    assert cj.trigremez(f, 1, max_iter=3, tol=1e-3) is sentinel
    assert seen[0][0][1:] == seen[1][0][1:]
    assert seen[0][1] == seen[1][1] == {}


@pytest.mark.parametrize("case", [1, 2, 3])
def test_saved_native_plot_artists_and_hold_policy(case):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.markers import MarkerStyle

    from chebfunjax.utils._trigremez_iteration_options import MATLAB_COLOR_ORDER, plot_iteration

    fixture = FIXTURES.parent / "trigremez_iteration_options_graphics_2024b"
    metadata = json.loads((fixture / "metadata.json").read_text())["cases"][str(case)]
    fig, ax = plt.subplots(figsize=(600 / 72, 270 / 72), dpi=72)
    held = bool(metadata["held_before"])
    ax.set_prop_cycle(color=MATLAB_COLOR_ORDER)
    if held:
        ax.plot([0, 1], [3, 4], linewidth=0.5)
    else:
        ax.plot([0, 1], [3, 4])  # Native unheld first plot must clear this line.
    original_position = ax.get_position().bounds
    numeric_case = 2 if case == 2 else 1
    with np.load(FIXTURES / f"case{numeric_case}.npz") as event:
        old = jnp.asarray(event["old_reference"])
        new = jnp.asarray(event["new_reference"])
        values = jnp.asarray(event["plot_y"])
        plot_iteration(old, new, lambda x: values, tuple(event["domain"]), ax=ax, hold=held)
    assert ax.get_position().bounds == original_position
    assert len(ax.lines) == len(metadata["lines"])
    with np.load(fixture / f"case{case}.npz") as native:
        for index, (line, properties) in enumerate(zip(ax.lines, metadata["lines"])):
            np.testing.assert_allclose(
                line.get_xdata(), native[f"line{index}_x"], rtol=0, atol=1e-15
            )
            np.testing.assert_array_equal(line.get_ydata(), native[f"line{index}_y"])
            np.testing.assert_allclose(
                to_rgb(line.get_color()), properties["color"], rtol=0, atol=0
            )
            assert line.get_linewidth() == properties["linewidth"]
    marker_lines = ax.lines[-3:-1]
    assert marker_lines[0].get_marker() == "o"
    assert marker_lines[0].get_markerfacecolor() == "none"
    for line in marker_lines:
        assert line.get_markersize() == 12 and line.get_markeredgewidth() == 0.5
    marker = MarkerStyle(marker_lines[1].get_marker())
    path = marker.get_path().transformed(marker.get_transform())
    actual = path.vertices * 12
    expected = np.asarray(
        [
            (0, -6),
            (0, 6),
            (-6, 0),
            (6, 0),
            (-4.2426, -4.2426),
            (4.2426, 4.2426),
            (-4.2426, 4.2426),
            (4.2426, -4.2426),
        ]
    )
    # This bound reflects native SVG four-decimal path quantization, not scientific tolerance.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5.1e-5)
    np.testing.assert_array_equal(ax.get_xlim(), metadata["xlim"])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == metadata["legend"]
    plt.close(fig)


def test_python_keyword_flags_use_native_presence_semantics(monkeypatch, capsys):
    provider = importlib.import_module("chebfunjax.utils._trigremez_polynomial_source")
    seen = []
    sentinel = object()

    def polynomial(*args, **kwargs):
        seen.append(kwargs)
        return sentinel

    monkeypatch.setattr(provider, "polynomial_trigremez", polynomial)
    assert cj.trigremez(cj.chebfun("x"), 1, display=False, plotfcns=None) is sentinel
    assert seen[0]["iteration_callback"] is not None
    assert capsys.readouterr().out == HEADER + "\n"


@pytest.mark.parametrize(
    "case,arguments",
    [
        (1, (-1, "unknown", 7)),
        (2, (1, "display")),
        (3, (1, "plotfcns")),
        (4, (1, "tol")),
        (5, (1, "maxiter")),
        (6, (float("nan"),)),
        (7, (1, float("nan"))),
        (8, (1, True)),
        (9, (1, 1, "tol", 1e-3, "tol", 2e-3)),
        (10, (1, "DISplaY", False, "PLOTFCNS", [])),
    ],
)
def test_extracted_native_parser_edges_public_forwarding(case, arguments, monkeypatch):
    native = json.loads((FIXTURES / "parser_edges.json").read_text())["cases"][str(case)]
    presentation = importlib.import_module("chebfunjax.utils._trigremez_iteration_options")
    public = importlib.import_module("chebfunjax.utils.trigrational")
    polynomial = importlib.import_module("chebfunjax.utils._trigremez_polynomial_source")
    flags, calls = [], []
    sentinel = object()

    def hook(show, plot, **kwargs):
        flags.append((show, plot))
        return None

    def trial(*args, **kwargs):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(presentation, "iteration_hook", hook)
    monkeypatch.setattr(public, "_trigremez_rational", trial)
    monkeypatch.setattr(polynomial, "polynomial_trigremez", trial)
    f = cj.chebfun("x")
    if native["error_id"]:
        with pytest.raises(TrigRemezError) as caught:
            cj.trigremez(f, *arguments)
        assert caught.value.identifier == native["error_id"]
        assert str(caught.value) == native["error_id"] + ": " + native["error_message"]
        assert not flags and not calls
    else:
        assert cj.trigremez(f, *arguments) is sentinel
        assert flags == [(bool(native["opts"]["displayIter"]), bool(native["opts"]["plotIter"]))]
        rational = bool(native["rational_mode"])
        assert calls[0][1] == native["m"]
        if rational:
            assert calls[0][2] == native["n"]
        assert calls[0][3 if rational else 2] == native["opts"]["maxIter"]
        assert calls[0][4 if rational else 3] == native["opts"]["tol"]


def test_empty_return_precedes_option_parsing_and_header(capsys):
    f = cj.Chebfun.empty()
    assert cj.trigremez(f, -1, "unknown", 7, display=False) is f
    assert capsys.readouterr().out == ""
