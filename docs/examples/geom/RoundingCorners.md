# Rounding corners by convolution

*Nick Trefethen, October 2011*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/geom/RoundingCorners.html)

(Chebfun example geom/RoundingCorners.m)

A piecewise-linear function with two corners:

![RoundingCorners figure 1](../../images/geom/RoundingCorners_repl_01.png)

Convolving with a narrow hat function of width $0.2$ (a mollifier)

![RoundingCorners figure 2](../../images/geom/RoundingCorners_repl_02.png)

rounds the corners into $C^1$ parabolic arcs:

![RoundingCorners figure 3](../../images/geom/RoundingCorners_repl_03.png)

The same trick applies to a planar curve, convolving each coordinate:

![RoundingCorners figure 4](../../images/geom/RoundingCorners_repl_04.png)

![RoundingCorners figure 5](../../images/geom/RoundingCorners_repl_05.png)

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
