"""Supplemental AAAApprox warning-channel adapter contracts, not native predicates."""
import ast
import warnings
from pathlib import Path

import pytest


@pytest.fixture
def adapter():
    source = Path(__file__).resolve().parents[2] / "examples/approx/aaa_approx.py"
    tree = ast.parse(source.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_aaa_with_output_warning")
    namespace = {"warnings": warnings}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"),
         namespace)
    return namespace


def test_actual_message_routed_once_and_result_preserved(adapter, capsys, monkeypatch):
    value = object()
    seen = []
    forwarded = []
    def worker(f):
        seen.append(f)
        warnings.warn("Function not resolved using 321 pts.", UserWarning)
        return value
    callback = object()
    adapter["_aaa_on_interval"] = worker
    monkeypatch.setattr(warnings, "showwarning", lambda *args: forwarded.append(args))
    assert adapter["_aaa_with_output_warning"](callback) is value
    assert seen == [callback]
    assert capsys.readouterr().out == "Warning: Function not resolved using 321 pts.\n"
    assert not forwarded


def test_resolved_call_does_not_invent_warning(adapter, capsys):
    value = object()
    adapter["_aaa_on_interval"] = lambda f: value
    assert adapter["_aaa_with_output_warning"](object()) is value
    output = capsys.readouterr()
    assert output.out == output.err == ""


def test_other_warning_category_location_and_message_preserved(adapter, capsys,
                                                               monkeypatch):
    forwarded = []
    def worker(f):
        warnings.warn_explicit("AAA cleanup: 1 Froissart doublet removed.",
                               RuntimeWarning, "source_callback.py", 73)
        return 7
    adapter["_aaa_on_interval"] = worker
    monkeypatch.setattr(warnings, "showwarning", lambda *args: forwarded.append(args))
    assert adapter["_aaa_with_output_warning"](object()) == 7
    assert capsys.readouterr().out == ""
    assert len(forwarded) == 1
    assert str(forwarded[0][0]) == "AAA cleanup: 1 Froissart doublet removed."
    assert forwarded[0][1:] == (RuntimeWarning, "source_callback.py", 73)
