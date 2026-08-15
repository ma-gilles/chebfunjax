"""Port of MATLAB Chebfun tests/treeVar/test_printTree.m (Fable 5).

Provenance
----------
MATLAB source : tests/treeVar/test_printTree.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.operators.treevar import TreeVar, print_tree


class TestTreevarPrintTree:
    def test_basic_computation(self):
        u = TreeVar()
        t = u.cos() + u.sin()
        s = print_tree(t.tree)
        assert isinstance(s, str) and "plus" in s
        s2 = t.print()
        assert isinstance(s2, str) and "cos" in s2

    def test_with_differentiation(self):
        u = TreeVar()
        t = 2 + u.diff(2)
        s = print_tree(t.tree)
        assert isinstance(s, str) and "diff" in s
        s2 = t.print()
        assert "numerical" in s2
