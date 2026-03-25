# The FFT in Chebfun

*Mark Richardson, May 2011*

*Original: [chebfun.org/examples/cheb/ChebfunFFT](https://www.chebfun.org/examples/cheb/ChebfunFFT.html)*

---

One of the most fundamental operations in Chebfun is converting between
**function values at Chebyshev points** and **Chebyshev expansion coefficients**.
This is achieved by the FFT in $O(n\log n)$ operations.

## Why the FFT works

The Chebyshev points on $[-1,1]$,

$$x_k = \cos\!\left(\frac{k\pi}{n}\right), \quad k=0,1,\ldots,n,$$

can be interpreted as the **real parts of equispaced nodes on the unit circle**.
A truncated Chebyshev expansion is equivalent to a truncated Laurent series,
and the Laurent coefficients can be extracted by the FFT.

## The connection explicitly

Let $f$ have values $\{f_k\}$ at the Chebyshev points (in ascending order).
Then:

1. Mirror: form $v = [f_{n}, f_{n-1}, \ldots, f_0, f_1, \ldots, f_{n-1}]$ — a
   vector of length $2n$.
2. FFT: compute $c = \text{Re}(\text{FFT}(v)) / n$.
3. Scale endpoints: $c_0 \leftarrow c_0/2$, $c_n \leftarrow c_n/2$.

```python
import numpy as np
import chebfunjax as cj
import jax.numpy as jnp

def cheb2coeffs_fft(fvals):
    """Convert ascending Chebyshev node values to coefficients via FFT."""
    n = len(fvals)
    vals_circle = np.concatenate([fvals[::-1], fvals[1:-1]])
    c_full = np.real(np.fft.fft(vals_circle)) / (n - 1)
    c = c_full[:n].copy()
    c[0] /= 2
    c[-1] /= 2
    return c
```

## Verification

```python
fc = cj.chebfun(lambda x: jnp.exp(x) * jnp.sin(jnp.pi * x) + x)
n = len(fc)
# Chebyshev-2 points (descending)
cheb_pts = np.cos(np.pi * np.arange(n) / (n-1))
fvals_asc = np.exp(cheb_pts[::-1]) * np.sin(np.pi * cheb_pts[::-1]) + cheb_pts[::-1]

coeffs_fft = cheb2coeffs_fft(fvals_asc)
coeffs_cj  = np.array(fc.coeffs)
print(f"Max difference: {np.max(np.abs(coeffs_fft - coeffs_cj)):.2e}")
```

```
Max difference: 4.44e-16
```

Machine precision. The FFT route and the internal chebfunjax representation
agree to rounding error.

![Values at Chebyshev nodes and coefficient decay](../../../images/cheb/chebfun_fft.png)

## O(n log n) complexity

The FFT achieves $O(n\log n)$ complexity for both directions:

- **Values → Coefficients**: FFT of the mirrored values.
- **Coefficients → Values**: Inverse FFT (equivalent to evaluating the
  Chebyshev expansion at all Chebyshev nodes simultaneously).

This makes chebfunjax fast even for functions requiring thousands of
coefficients. For $n = 10^6$, the FFT takes under a second on modern hardware.

## References

1. J. P. Boyd, *Chebyshev and Fourier Spectral Methods*, Dover, 2001.
2. L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000, Ch. 3.
