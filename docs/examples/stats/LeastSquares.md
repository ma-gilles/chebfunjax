# Discrete and continuous least squares

*Alex Townsend, March 2013*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/stats/LeastSquares.html)

(Chebfun example stats/LeastSquares.m)

`polyfit` fits noisy samples of the Runge function by a degree-10
polynomial in the least-squares sense:

![LeastSquares figure 1](../../images/stats/LeastSquares_repl_01.png)

The continuous analogue fits a piecewise-smooth chebfun by its best
degree-10 polynomial in $L^2$:

![LeastSquares figure 2](../../images/stats/LeastSquares_repl_02.png)

(The noise draw is our own; the constructions replicate.)

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
