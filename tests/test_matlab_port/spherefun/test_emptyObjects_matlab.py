"""Port of MATLAB ``tests/spherefun/test_emptyObjects.m`` (Fable 5).

The original test executes all commands inside one ``try/catch`` and checks
only that none raises. This port keeps every original command and records the
Python API adapter next to it; it intentionally does not assert output values
or types except for the separately source-defined ``minandmax2`` empty pair.

Provenance
----------
MATLAB source : tests/spherefun/test_emptyObjects.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.spherefun.spherefun import Spherefun


@pytest.mark.parametrize(
    "source_expression,operation",
    [
        ("f + f", lambda f: f + f),
        ("2 * f", lambda f: 2 * f),
        ("f * 2", lambda f: f * 2),
        ("f .^ 2", lambda f: f ** 2),
        ("2 .^ f", lambda f: 2 ** f),
        ("f .^ f", lambda f: f ** f),
        ("sqrt(f)", lambda f: f.sqrt()),
        ("sum(f)", lambda f: f.sum()),
        ("integral2(f)", lambda f: f.integral2()),
        ("norm(f)", lambda f: f.norm()),
        ("squeeze(f)", lambda f: f.squeeze()),
        ("diff(f)", lambda f: f.diff()),
        ("cos(f)", lambda f: f.cos()),
        ("sin(f)", lambda f: f.sin()),
        ("sinh(f)", lambda f: f.sinh()),
        ("f .^ f + f", lambda f: f ** f + f),
        ("diag(f)", lambda f: f.diag()),
        ("trace(f)", lambda f: f.trace()),
        ("mean(f)", lambda f: f.mean()),
        ("minandmax2(f)", lambda f: f.minandmax2()),
        # The MATLAB test discards all outputs; make that explicit for the
        # empty median source branch, which leaves its output unassigned.
        ("median(f) [nargout=0]", lambda f: f.median(nargout=0)),
        ("flipud(f)", lambda f: f.flipud()),
        ("flipdim(f, 1)", lambda f: f.flipdim(1)),
    ],
)
def test_every_original_empty_command_does_not_raise(source_expression, operation):
    f = Spherefun.empty()
    operation(f)


def test_minandmax2_empty_result_is_explicitly_empty():
    # Unlike max2/min2's source early return, minandmax2 explicitly assigns
    # both outputs to [] in the pinned MATLAB implementation.
    values, locations = Spherefun.empty().minandmax2()
    assert values.shape == (0,)
    assert locations.shape == (0,)


def test_empty_sum2_is_zero():
    # @spherefun/sum2.m contracts no factors to scalar zero.
    assert float(Spherefun.empty().sum2()) == 0.0


@pytest.mark.parametrize("method", ["max2", "min2"])
def test_empty_max2_min2_requested_outputs_match_native_unassigned_error(method):
    with pytest.raises(ValueError, match="MATLAB:unassignedOutputs"):
        getattr(Spherefun.empty(), method)()


def test_empty_median_requested_output_matches_native_unassigned_error():
    with pytest.raises(ValueError, match="MATLAB:unassignedOutputs"):
        Spherefun.empty().median()


class TestSupplementalEmptyOutputs:
    def test_all_commands_tolerate_empty(self):
        f = Spherefun.empty()
        assert f.norm().shape == (0,)
        results = [f + f, f * 2, f ** 2, -f, f.laplacian(), f.cos()]
        # @spherefun/sum2.m returns scalar zero when idxPlus is empty.
        assert float(f.sum2()) == 0.0
        for r in results:
            assert hasattr(r, "isempty") and r.isempty()
