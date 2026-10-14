"""Source warning semantics for MATLAB @chebfun/vertcat (7574c77)."""
import warnings

import numpy as np

from chebfunjax import chebfun
from chebfunjax.chebfun1d import _vertcat
from chebfunjax.chebfun1d.chebfun import Chebfun


def _row(f):
    return Chebfun.horzcat(np.zeros((1, 0)), f, np.zeros((1, 1)))


def _concat_first(a, b):
    return Chebfun.vertcat(a, b)


def _concat_second(a, b):
    return Chebfun.vertcat(a, b)


def _warning_text():
    return (
        "CHEBFUN:CHEBFUN:vertcat:join: "
        "Vertical concatenation of CHEBFUN objects now produces a CHEBMATRIX\n"
        "The V4 behaviour can be reproduced using the JOIN() method."
    )


def test_vertcat_warning_is_once_only_and_reset_is_test_scoped(monkeypatch):
    x = chebfun(lambda t: t, domain=(0.0, 1.0))
    row1, row2 = _row(x), _row(x + 1)

    # Simulate a preceding test or operation consuming the process-global warning.
    monkeypatch.setattr(_vertcat, "_JOIN_WARNING_ENABLED", True)
    with warnings.catch_warnings(record=True) as prior:
        warnings.simplefilter("always")
        _concat_first(row1, row2)
    assert [str(item.message) for item in prior] == [_warning_text()]
    assert _vertcat._JOIN_WARNING_ENABLED is False

    # Explicitly re-enable only inside this test, then verify once-only behavior.
    monkeypatch.setattr(_vertcat, "_JOIN_WARNING_ENABLED", True)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        first = _concat_first(row1, row2)
        second = _concat_second(row2, row1)
    assert [str(item.message) for item in seen] == [_warning_text()]
    assert first.size == (2, 2)
    assert second.size == (2, 2)
    assert first.domain == second.domain == x.domain.breakpoints
