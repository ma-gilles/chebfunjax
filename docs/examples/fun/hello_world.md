# Hello World

**Original:** [fun/HelloWorld](https://www.chebfun.org/examples/fun/HelloWorld.html)
**Author(s):** Alex Townsend, March 2013

---

In any programming language, printing "Hello World" is always a first example.
Here we display "HELLO" using Chebfun2, demonstrating how low-rank bivariate
functions can encode text.

## A matrix encoding of HELLO

A $15 \times 40$ binary matrix $A$ encodes the five letters of "HELLO" as
blocks of ones on a zero background (from Exercise 9.3 of [1]). The matrix
has rank 10 because five of its rows are entirely zero.

## Constructing a chebfun2 from discrete data

Usually Chebfun2 is passed a function of two variables, but it can also deal
with discrete data. The matrix $A$, of size $m \times n$, is assumed to contain
data sampled on an $m \times n$ Chebyshev tensor grid, and the resulting
chebfun2 interpolates $A$:

$$f = \text{chebfun2}(A), \qquad \|A - f(\text{chebpts})\| \approx 0.$$

## Saying Hello at different ranks

The Chebfun2 constructor can also be given an integer $k$ so that the
resulting object has rank exactly $k$. Contour plots of the rank-$k$
approximations for $k = 1, 3, 5, 7, 10$ show the word "HELLO" emerging
from blurry blobs into sharp lettering as the rank increases.


![Hello World](../../images/fun/hello_world.png)

1. L. N. Trefethen and D. Bau III, *Numerical Linear Algebra*, SIAM, 1997.

## Code

```python
import numpy as np

A = np.zeros((15, 40))
A[1:9, 1:3] = 1; A[4:6, 3:5] = 1; A[1:9, 5:7] = 1
A[2:10, 9:11] = 1; A[2:4, 9:15] = 1; A[5:7, 9:15] = 1
A[8:10, 9:15] = 1; A[3:11, 17:19] = 1; A[9:11, 17:24] = 1
A[4:12, 25:27] = 1; A[10:12, 25:31] = 1
A[5:13, 33:35] = 1; A[5:13, 37:39] = 1
A[5:7, 35:37] = 1; A[12:13, 35:37] = 1
print(f"rank of the HELLO matrix: {np.linalg.matrix_rank(A)}")
```


## References

## Figures (chebfun.org parity)

![HelloWorld figure 1](../../images/fun/HelloWorld_01.png)

![HelloWorld figure 2](../../images/fun/HelloWorld_02.png)

![HelloWorld figure 3](../../images/fun/HelloWorld_03.png)

![HelloWorld figure 4](../../images/fun/HelloWorld_04.png)

![HelloWorld figure 5](../../images/fun/HelloWorld_05.png)

![HelloWorld figure 6](../../images/fun/HelloWorld_06.png)
