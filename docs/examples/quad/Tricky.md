# Some tricky integrals

*Fredrik Johansson and Nick Trefethen, March 2018*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/quad/Tricky.html)

(Chebfun example quad/Tricky.m)

## 0. Introduction

FJ gave a talk at ENS Lyon today with a number of examples in it that
intrigued LNT.  Here we play with some of those examples in Chebfun.
Now Chebfun is just numerical, with no guarantees of accuracy, which
means it is not a competitor for FJ's Arb method of rigorous quadrature
[1].  The point of this example is only to see how Chebfun does,
unrigorously in floating point arithmetic, on some challenging examples
people have cooked up over the years.

The published MATLAB run finds a highly accurate answer in all but one
example; its Example 3 loses ten digits.  This replica reproduces every
integral, and — notably — does *not* lose accuracy on Example 3.

## 1. Three spikes

From Cranley & Patterson (1971) and the Kahaner battery; see also
[quad/SpikeIntegral](SpikeIntegral.md):

```
Iexact =
   0.210802735500549
I =
   0.210802735500549
```

## 2. Violent oscillation

$\int_0^8 \sin(x+e^x)\,dx$, from Rump's *Verification methods* (2010):

```
Iexact =
   0.347400172657248
I =
   0.347400172657248
```

(The published MATLAB value is `0.347400172657246`; this replica's
answer agrees with the exact value in all 15 digits.)

## 3. Violent oscillation with 2979 discontinuities

$\int_0^8 (e^x-\lfloor e^x\rfloor)\sin(x+e^x)\,dx$.  This is where the
published MATLAB run fails: it reports `I = 0.087881488553783` against
the exact `0.098651704478365` — an error of $10^{-2}$ — and after
raising `splitMaxLength` to $10^6$ it still gets only 6 digits after
72 seconds, prompting the authors to remark "for it to lose ten digits
of accuracy looks like a bug somewhere."

This replica constructs the integrand on the 2979 known breakpoints
$\log 2, \log 3, \dots$ (obtained from `floor` of the exponential
chebfun) and obtains

```
Iexact =
   0.098651704478365
I =
   0.098651704393442
```

— accurate to $8.5\times 10^{-11}$, ten digits better than the
published MATLAB result.

## 4. Error function

From Silviu Filip: $\int_{-1}^1 e^{-x}\,\mathrm{erf}(\sqrt{1250}x+1.5)\,dx$:

```
I =
   -0.999065350291922
```

(Digit-for-digit with the published value.)

## 5. Airy function

From FJ: $\int_0^\infty \mathrm{Ai}(x)\,dx$ on the unbounded domain,

```
Iexact =
   0.378751605379087
I =
   0.378751605379087
```

and on the truncated interval $[0,40]$:

```
I =
   0.378751605379086
```

## 6. Absolute value of polynomial

Helfgott's MathOverflow integral
$\int_0^1 |(x^4+10x^3+19x^2-6x-6)|\,e^x dx$:

```
Iexact =
  11.147310550057140
I =
  11.147310550057584
```

(Published: `11.147310550057142`.)

## 7. A ceiling function

Gauss's schoolboy sum as an integral,
$\int_0^{10} \lceil x \rceil\, x\, dx$-type representation:

```
Iexact =
        5050
I =
        5050
```

## 8. Another non-smooth function

```
Iexact =
  -0.142818642026328
I =
  -0.142818642026306
```

(Published: `-0.142818642026329`.)

## 9. From Brisebarre and Joldes

$\int_0^3 \sin(10^5 x^4)$-type wild oscillation (Chen 2006; Joldes's
thesis), requiring a representation of enormous length:

```
Iexact =
   0.749974368527195
I =
   0.749974368527170
```

(Published: `0.749974368527190` global / `0.749974368527184` with
splitting.)

## Reference

[1] F. Johansson, Numerical integration in arbitrary-precision ball
arithmetic, arXiv:1802.07942 and _International Congress on
Mathematical Software_, Springer, Cham, 2018.

---

*Replicated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); original
example copyright The University of Oxford and The Chebfun Developers.*
