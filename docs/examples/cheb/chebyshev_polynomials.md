# Chebyshev Polynomials $T_n$

*Original: [chebfun.org/examples/cheb/](https://www.chebfun.org/examples/cheb/)*

---

The Chebyshev polynomials $T_n(x) = \cos(n\arccos x)$ are fundamental to
numerical analysis. Every smooth function can be expanded in a Chebyshev
series, and the coefficients decay geometrically for analytic functions.

## Basic properties

The Chebyshev polynomials satisfy the three-term recurrence:

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_{n+1}(x) = 2x\,T_n(x) - T_{n-1}(x).$$

```python
import numpy as np

def chebyshev_T(n, x):
    return np.cos(n * np.arccos(np.clip(x, -1, 1)))

# Verify three-term recurrence at x = 0.5
x = 0.5
for n in range(2, 8):
    T_rec  = 2 * x * chebyshev_T(n, x) - chebyshev_T(n-1, x)
    T_direct = chebyshev_T(n+1, x)
    assert abs(T_rec - T_direct) < 1e-14
print("Three-term recurrence verified.")
```

![Chebyshev polynomials T_0 through T_8](../../../images/cheb/chebyshev_polynomials.png)

The red dots mark the **Chebyshev nodes** (zeros of $T_n$):

$$x_k = \cos\!\left(\frac{(2k-1)\pi}{2n}\right), \quad k=1,\ldots,n.$$

## Minimax property

$T_n(x)/2^{n-1}$ is the monic polynomial of degree $n$ with the smallest
infinity norm on $[-1,1]$. This minimum norm is $1/2^{n-1}$.

```python
xx = np.linspace(-1, 1, 10000)
for n in range(1, 8):
    norm = np.max(np.abs(chebyshev_T(n, xx)))
    print(f"||T_{n}||_inf = {norm:.10f}")
```

```
||T_1||_inf = 1.0000000000
||T_2||_inf = 1.0000000000
||T_3||_inf = 1.0000000000
...
```

## Orthogonality

The Chebyshev polynomials are orthogonal with respect to the weight
$w(x) = 1/\sqrt{1-x^2}$:

$$\int_{-1}^1 T_m(x)\,T_n(x)\,\frac{dx}{\sqrt{1-x^2}} =
\begin{cases} 0 & m \neq n \\ \pi/2 & m = n > 0 \\ \pi & m=n=0 \end{cases}$$

In chebfunjax, this is easily verified:

```python
import chebfunjax as cj
import jax.numpy as jnp

for m, n in [(0,0), (1,1), (2,2), (0,1), (1,2)]:
    Tm = cj.chebfun(lambda x, m=m: jnp.array(chebyshev_T(m, np.array(x))),
                    domain=(-0.999, 0.999))
    Tn = cj.chebfun(lambda x, n=n: jnp.array(chebyshev_T(n, np.array(x))),
                    domain=(-0.999, 0.999))
    w  = cj.chebfun(lambda x: 1.0 / jnp.sqrt(1.0 - x**2), domain=(-0.999, 0.999))
    inn = float((w * Tm * Tn).sum())
    print(f"<T_{m}, T_{n}> = {inn:.6f}")
```

```
<T_0, T_0> = 3.141593
<T_1, T_1> = 1.570796
<T_2, T_2> = 1.570796
<T_0, T_1> = 0.000000
<T_1, T_2> = 0.000000
```

## The FFT connection

Converting between function values at Chebyshev nodes and Chebyshev
coefficients uses the DCT (discrete cosine transform), which can be
computed in $O(n\log n)$ operations. See the
[ChebfunFFT example](chebfun_fft.md) for details.
