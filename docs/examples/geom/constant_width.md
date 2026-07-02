# Polynomial Level Curve of Constant Width

**Original:** [geom/ConstantWidth](https://www.chebfun.org/examples/geom/ConstantWidth.html)
**Author(s):** Nick Trefethen, May 2022

---

A _Mathematics Today_ column by Alan Champneys [2] describes a fascinating
example that can be found in Wikipedia [4] and goes back to a paper by
Rabinowitz [3]; see also [1].

## The Rabinowitz polynomial

Consider the bivariate polynomial $p(x,y)$ defined by

$$
p(x,y) = r^8 - 45r^6 - 41283r^4 + 7950960r^2
+ 16q^3 + 48r^2 q^2 + xq(16r^4 - 5544r^2 + 266382) - 373248000,
$$

where $r^2 = x^2 + y^2$ and $q = x^2 - 3y^2$.

The remarkable result is that the zero set of $p$ has **constant width** in
every direction in the plane -- like the British 50p coin.

## Verifying constant width

The zero-level curve can be computed and its width measured in several
directions. For five equally-spaced angles the widths agree to about 5
digits, which is not bad considering the size of the coefficients. The
exact width is 18, as can be verified by setting $y = 0$, which reduces
the polynomial to a univariate polynomial that vanishes at $x = -8$ and
$x = 10$.

## Perimeter

Just for fun one can also compute the perimeter of the "coin", presumably
accurate to about 5 digits.

## References

1. M. Bardet and T. Bayen, On the degree of the polynomial defining a
   planar algebraic curve of constant width, arXiv:1312.4358v1, 2013.

2. A. Champneys, Westward Ho! Musing on mathematics and mechanics,
   _Mathematics Today_, April 2022, 56--59.

3. S. Rabinowitz, A polynomial curve of constant width, _Missouri Journal
   of Mathematical Sciences_ 9 (1997), 23--27.

4. Wikipedia, "Curve of constant width".


![Constant Width Curves](../../images/geom/constant_width.png)

## Code

```python
from examples.geom.constant_width import run
run()
```
