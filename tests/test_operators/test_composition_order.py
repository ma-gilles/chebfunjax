"""Composition order metadata, native adchebfun.compose/operatorBlock.plus.

Chebfun commit: 7574c77; independent and state-dependent source chain rules.
"""
from chebfunjax.operators.chebop import _EqOrderSniffer, _SysOrderSniffer


def test_system_composition_orders():
    orders = [0, 0]
    u = _SysOrderSniffer(orders, (0,))
    v = _SysOrderSniffer(orders, (1,))
    t = _SysOrderSniffer(orders, ())
    assert u(t)._depths == {0: 0}
    assert u(0., "right")._depths == {0: 0}
    assert u(v.diff(2)).diff()._depths == {0: 1, 1: 3}
    assert orders == [1, 3]


def test_equation_composition_orders():
    u = _EqOrderSniffer({(0, 0)})
    v = _EqOrderSniffer({(1, 2)})
    assert u(v).diff().pairs == frozenset({(0, 1), (1, 3)})
    assert u(0., "left").pairs == u.pairs
