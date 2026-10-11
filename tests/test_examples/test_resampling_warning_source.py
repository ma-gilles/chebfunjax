"""Supplemental source-toggle output adapters for the resampling example."""
import ast
import warnings
from pathlib import Path

import pytest

import chebfunjax.chebpref as chebpref


@pytest.fixture
def adapter(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "examples/stats/resampling_random_variables.py"
    tree = ast.parse(source.read_text())
    helper = next(x for x in tree.body if isinstance(x, ast.FunctionDef)
                  and x.name == "_splitting_with_output_warning")
    namespace = {"warnings": warnings, "splitting": chebpref.splitting}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"),
         namespace)
    saved = chebpref.ChebfunPref._defaults
    monkeypatch.setattr(chebpref, "_SPLITTING_WARNING_EMITTED", False)
    try:
        chebpref.ChebfunPref.setDefaults("factory")
        yield namespace
    finally:
        chebpref.ChebfunPref._defaults = saved


def test_source_off_on_routes_actual_warning_once(adapter, capsys):
    toggle = adapter["_splitting_with_output_warning"]
    assert toggle("off") == "off"
    assert chebpref.ChebfunPref().splitting is False
    assert toggle("on") == "off"
    assert chebpref.ChebfunPref().splitting is True
    assert chebpref._SPLITTING_WARNING_EMITTED
    output = capsys.readouterr()
    assert output.out == "Warning: " + chebpref._SPLITTING_WARNING_TEXT + "\n"
    assert output.err == ""


def test_previously_emitted_warning_is_not_recreated(adapter, capsys, monkeypatch):
    monkeypatch.setattr(chebpref, "_SPLITTING_WARNING_EMITTED", True)
    toggle = adapter["_splitting_with_output_warning"]
    toggle("off")
    toggle("on")
    output = capsys.readouterr()
    assert output.out == output.err == ""
    assert chebpref._SPLITTING_WARNING_EMITTED
    assert chebpref.ChebfunPref().splitting is True


def test_other_warnings_keep_message_category_location(adapter, capsys, monkeypatch):
    forwarded = []
    def callback(state):
        warnings.warn_explicit("other actual warning", RuntimeWarning,
                               "source_callback.py", 37)
        return "off"
    adapter["splitting"] = callback
    monkeypatch.setattr(warnings, "showwarning", lambda *args: forwarded.append(args))
    assert adapter["_splitting_with_output_warning"]("off") == "off"
    assert capsys.readouterr().out == ""
    assert len(forwarded) == 1
    assert str(forwarded[0][0]) == "other actual warning"
    assert forwarded[0][1:] == (RuntimeWarning, "source_callback.py", 37)
