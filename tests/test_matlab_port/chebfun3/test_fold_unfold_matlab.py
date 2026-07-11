"""Port of MATLAB Chebfun tests/chebfun3/test_fold_unfold.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_fold_unfold.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="tensor fold/unfold are MATLAB-internal helpers")


class TestChebfun3Foldunfold:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
