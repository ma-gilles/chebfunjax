# Fourier-based chebfuns

*Grady Wright, June 2014*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/fourier/FourierBasedChebfuns.html)

Python translation: [`examples/fourier/fourier_based_chebfuns.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/fourier/fourier_based_chebfuns.py)

One of the new features of Chebfun version 5 is the ability to create chebfuns of smooth periodic functions using Fourier series. This example introduces and demonstrates some of the functionality of this new tool.

## Construction and comparison

Fourier-based chebfuns, or "trigfuns" as we like to refer to them, can be created with the use of the `'trig'` flag in the chebfun constructor. For, example, the function $f(x) = \cos(8\sin(x))$ for $-\pi \leq x \leq \pi$ can be constructed as follows:

```matlab
dom = [-pi,pi];
f = chebfun(@(x) cos(8*sin(x)),dom,'trig')
plot(f);
```

```text
f =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[    -3.1,     3.1]       61         1        1
vertical scale =   1
```

![FourierBasedChebfuns figure 01](../../images/fourier/FourierBasedChebfuns_01.png)

Here $f$ is represented to machine precision using a Fourier interpolant rather than a Chebyshev interpolant. The displayed information for $f$ above shows that it is of length 61, meaning that $f$ is resolved to machine precision using 61 samples, or $(61-1)/2=30$ (complex) Fourier modes. These coefficients can be displayed graphically by

```matlab
plotcoeffs(f), ylim([1e-18 1])
```

![FourierBasedChebfuns figure 02](../../images/fourier/FourierBasedChebfuns_02.png)

Since $f$ is smooth and periodic, a Fourier representation requires fewer terms than a Chebyshev representation of $f$ to reach machine precision. We can check this by constructing $f$ without the `'trig'` flag:

```matlab
f_cheby = chebfun(@(x) cos(8*sin(x)),dom)
```

```text
f_cheby =
   chebfun column (1 smooth piece)
       interval       length     endpoint values
[    -3.1,     3.1]      103         1        1
vertical scale =   1
```

The ratio of length of the Chebyshev series to the Fourier series should be approximately $\pi/2$ since the former has a resolution power of $\pi$ points per wavelength and the latter of 2 points per wavelength. We can check this numerically as

```matlab
ratio = length(f_cheby)/length(f)
theoretical = pi/2
```

```text
ratio =
   1.688525
theoretical =
   1.570796
```

Trying to construct a trigfun from a non-periodic or non-smooth function will typically result in a warning being issued and an "unhappy" trigfun, as illustrated for the unit step function below:

```matlab
f = chebfun(@(x) 0.5*(1+sign(x)),dom,'trig')
plot(f);
```

![FourierBasedChebfuns figure 03](../../images/fourier/FourierBasedChebfuns_03.png)

The length of $f$ is 65536, which is the maximum number of samples used in the construction process to try to resolve $f$. The famous Gibbs phenomenon can be seen near the discontinuity in the plot of $f$. Chebfun can be used to represent this function in non-periodic mode (i.e. using Chebyshev series) with the option of `splitting on`:

```matlab
f = chebfun(@(x) 0.5*(1+sign(x)),dom,'splitting','on')
```

```text
f =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[    -3.1,     3.1]    65536         0        0
vertical scale =   1
```

Splitting is not an option for trigfuns.

## Basic operations

Many Chebfun operations can also be applied directly to a trigfun. Some of these basic operations are illustrated in the examples below.

Addition, subtraction, multiplication, division, and function composition can all be directly applied to a trigfun. However one should be aware that operation should result in a smooth and periodic function. (If not, it will be converted to a nonperiodic chebfun.) The following example illustrates some of these operations:

```matlab
g = chebfun(@(x) sin(x),dom,'trig');
f = tanh(cos(1+2*g)^2)-0.5
plot(f)
```

```text
f =
   chebfun column (2 smooth pieces)
       interval       length     endpoint values
[    -3.1,2.2e-308]        1         0        0
[2.2e-308,     3.1]        1         1        1
vertical scale =   1    Total length = 2
f =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[    -3.1,     3.1]      161     -0.22    -0.22
vertical scale = 0.5
```

![FourierBasedChebfuns figure 04](../../images/fourier/FourierBasedChebfuns_04.png)

The max, min, and roots of $f$ can be computed by

```matlab
[maxf,xmaxf] = max(f);
[minf,xminf] = min(f);
rootsf = roots(f);
maxf
minf
rootsf
```

```text
maxf =
   0.261594
minf =
  -0.500000
rootsf =
  -3.009212
  -2.090420
  -1.051172
  -0.132380
   0.779312
   2.362280
```

These can be visualized as

```matlab
plot(f), hold on
plot(xmaxf,maxf,'gs',xminf,minf,'md',rootsf,0*rootsf,'ro')
legend('f','max f','min f','zeros f','location','southwest')
hold off;
```

![FourierBasedChebfuns figure 05](../../images/fourier/FourierBasedChebfuns_05.png)

The derivative of $f$ is computed using `diff`:

```matlab
df = diff(f);
plot(df)
```

![FourierBasedChebfuns figure 06](../../images/fourier/FourierBasedChebfuns_06.png)

and the definite integral is computed using `sum`:

```matlab
intf = sum(f)
```

```text
intf =
  -0.074011
```

Complex-valued trigfuns are also possible. For example:

```matlab
f = chebfun(@(x) 1i*(13*cos(x)-5*cos(2*x)-2*cos(3*x)-cos(4*x)) + ...
                 16*sin(x)^3, dom, 'trig')
plot(f), axis equal
```

```text
f =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[    -3.1,     3.1]        9     complex values
vertical scale =  17
```

![FourierBasedChebfuns figure 07](../../images/fourier/FourierBasedChebfuns_07.png)

The area enclosed by this curve can be computed as

```matlab
area_heart = abs(sum(real(f)*diff(imag(f))))
```

```text
area_heart =
  565.486678
```

According to [1], the true area enclosed is $180\pi$. The relative error in the computation above is then

```matlab
err = (area_heart - 180*pi)/(180*pi)
```

```text
err =
    -6.031274062627113e-16
```

The convolution of two smooth periodic functions can be computed using the `circconv` (circular convolution) function. The example below demonstrates this function in combination with the additional feature that allows trigfuns to be constructed from function values. The latter is demonstrated first:

```matlab
rng('default'), rng(0)
n = 201;
x = trigpts(n);
func_vals = exp(sin(2*pi*x)) + 0.05*randn(n,1);
f = chebfun(func_vals,dom,'trig')
```

```text
f =
   chebfun column (1 smooth piece)
       interval       length     endpoint values trig
[    -3.1,     3.1]      201      0.55     0.55
vertical scale = 2.8
```

Here $f$ interpolates the noisy `func_vals` at 201 equally spaced points from $[-\pi,\pi)$ using the Fourier basis. The high frequencies in this function can be smoothed by convolving it with a mollifier, in this case a (normalized) Gaussian with variance 0.1.

```matlab
sigma = 0.1;
g = chebfun(@(x) 1/(sigma*sqrt(2*pi))*exp(-0.5*(x/sigma)^2),dom,'trig');
```

Note that the resulting respresentation of $g$ is actually the periodic extension of the Gaussian over $[-\pi,\pi]$. The convolution of $f$ and $g$ is computed and visualized using

```matlab
h = circconv(f,g);
plot(g,'b'), hold on
plot(f,'r'), plot(h,'k')
legend('Mollifier g','Noisy function f','Smoothed function h');
hold off;
```

![FourierBasedChebfuns figure 08](../../images/fourier/FourierBasedChebfuns_08.png)

## References

1. Mathworld Heart Curve: [http://mathworld.wolfram.com/HeartCurve.html](http://mathworld.wolfram.com/HeartCurve.html)

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
