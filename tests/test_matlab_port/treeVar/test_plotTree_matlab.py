"""Port of MATLAB Chebfun tests/treeVar/test_plotTree.m (Fable 5).

Provenance
----------
MATLAB source : tests/treeVar/test_plotTree.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from chebfunjax.operators.treevar import TreeVar, plot_tree  # noqa: E402


class TestTreevarPlotTree:
    def test_basic_computation(self):
        import matplotlib.pyplot as plt
        u = TreeVar()
        t = u.cos() + u.sin()
        assert plot_tree(t.tree) is not None
        assert t.plot() is not None
        plt.close("all")

    def test_with_differentiation(self):
        import matplotlib.pyplot as plt
        u = TreeVar()
        s = 2 + u.diff(2)
        assert plot_tree(s.tree) is not None
        assert s.plot() is not None
        plt.close("all")
