"""Delta convolution storage: source @deltafun/conv.m, Chebfun 7574c77."""
import chebfunjax as cj


def test_order_addition_and_sorted_locations():
    zero = cj.chebfun(0, domain=[-1, 4])
    left = cj.Chebfun(funs=zero.funs, domain=zero.domain,
                      deltas=((0., 2., 1), (1., 3., 2)))
    right = cj.Chebfun(funs=zero.funs, domain=zero.domain,
                       deltas=((0., 5.), (3., 7., 1)))
    result = left.conv(right)
    assert result.deltas == ((0., 10., 1), (1., 15., 2),
                             (3., 14., 2), (4., 21., 3))
    assert result.sum() == 0
    assert left.deltas == ((0., 2., 1), (1., 3., 2))
    assert right.deltas == ((0., 5.), (3., 7., 1))
