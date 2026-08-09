# Random functions in 2D

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/approx2/Random2D.html)

(Chebfun example approx2/Random2D.m — Nick Trefethen, April 2017)

`randnfun2` constructs smooth random functions in 2D — continuous
analogues of `randn` built from finite Fourier series with
independent normally distributed random coefficients, with a space
scale parameter $\lambda$ (wave numbers up to about $2\pi/\lambda$).
(Draws here use seeded numpy streams; MATLAB's `rng(0)` stream is
not reproducible outside MATLAB.)

A random function with $\lambda = 0.2$ on a $2\times 1$ rectangle,
negative values black and positive white:

![Random2D figure 1](../../images/approx2/Random2D_repl_01.png)

A contour plot shows more:

![Random2D figure 2](../../images/approx2/Random2D_repl_02.png)

To isolate the zero contours to high accuracy (though it takes
longer), one can use `roots`:

![Random2D figure 3](../../images/approx2/Random2D_repl_03.png)

Here's a 3D plot:

![Random2D figure 4](../../images/approx2/Random2D_repl_04.png)

For comparison, a periodic random function (`'trig'`):

![Random2D figure 5](../../images/approx2/Random2D_repl_05.png)

And random functions with $\lambda = 0.1$:

![Random2D figure 6](../../images/approx2/Random2D_repl_06.png)

and with $\lambda = 0.05$:

![Random2D figure 7](../../images/approx2/Random2D_repl_07.png)

## Reference

1. S. Filip, A. Javeed, and L. N. Trefethen, Smooth random
   functions, random ODEs, and Gaussian processes, _SIAM Review_,
   61 (2019), 185-205.

---

*Replica script: [`examples/approx2/random2d_replica.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/approx2/random2d_replica.py).
Original example copyright by The University of Oxford and The Chebfun
Developers.*
