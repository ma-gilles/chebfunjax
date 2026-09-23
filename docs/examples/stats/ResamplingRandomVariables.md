# Sampling from a probability distribution

*Toby Driscoll, December 2011*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/stats/ResamplingRandomVariables.html)

Python translation: [`examples/stats/resampling_random_variables.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/stats/resampling_random_variables.py)

A common problem in applications of random variables is to draw samples from a given distribution. It's easy to find computer codes for generating pseudorandom numbers that are distributed uniformly or normally, and these usually must be converted to simulate a different target distribution. The key steps are integration and function inversion, which Chebfun can do with great accuracy.

```matlab
LW = 'linewidth'; FS = 'fontsize'; MS = 'markersize';
splitting off
```

```text
missing =
     2.449569436180354e-10
```

## von Mises distribution

The von Mises distribution is a periodic variant of the normal distribution. While the density is easily defined, it's otherwise not simple to work with analytically.

We start with the density function, normalized to give total probability 1.

```matlab
kappa = 1.5;
f = chebfun(@(x) exp(kappa*cos(x)),[-pi pi]);
density = f/sum(f);
```

Now we integrate to get the cumulative distribution function.

```matlab
cdf = cumsum(density);
plot([density,cdf],LW,1.6), axis([-pi pi 0 1])
title('von Mises distribution',FS,12)
legend('density','distribution','Location','northwest')
```

![ResamplingRandomVariables figure 01](../../images/stats/ResamplingRandomVariables_01.png)

Sampling from this distribution involves applying its inverse to uniformly sampled points. We could do this one-by-one using `roots`, but for a large number of points it is more efficient to find a chebfun for the inverse function with `inv`:

```matlab
cdfinv = inv(cdf);
plot(cdfinv,LW,1.6)
title('Inverse of von Mises distribution',FS,12)
```

![ResamplingRandomVariables figure 02](../../images/stats/ResamplingRandomVariables_02.png)

Now the resampling is easy. We compare the resulting histogram to the original von Mises density.

```matlab
u = rand(1e4,1);                           % uniform
x = cdfinv(u);                             % von Mises
[count,bin] = hist(x,36);
count = count/sum(count*(bin(2)-bin(1)));  % renormalize, total area = 1
cla, bar(bin,count), hold on
plot(density,'r',LW,1.6), axis tight
title('Sampled points and the orignal density',FS,12)
```

![ResamplingRandomVariables figure 03](../../images/stats/ResamplingRandomVariables_03.png)

## Logit-normal distribution

A more exotic and troublesome distribution is the logit-normal distribution. Its density and cdf are easy enough to define:

```matlab
sig = 1.11;
f = @(x) exp( -(log(x./(1-x))).^2/(2*sig^2))./(x.*(1-x));
density = chebfun(f,[0 1]);
density = density/sum(density);
cdf = cumsum(density);
clf, plot([density,cdf],LW,1.6)
title('logit-normal distribution',FS,12)
legend('density','distribution','Location','northwest')
```

![ResamplingRandomVariables figure 04](../../images/stats/ResamplingRandomVariables_04.png)

However, because $F'=f=0$ at the ends, the inverse function has infinite slope at the ends, and a straightforward inversion will fail. To cope with this, we'll take some shortcuts. First, we'll use symmetry to restrict attention to $x> 1/2$. Second, we'll put Chebfun into splitting mode to help cope with the endpoint slope. Finally, we'll truncate the domain of the cdf slightly.

```matlab
splitting on
cdfinv = inv( cdf{0.5,1-1e-3} );
clf, plot(cdfinv,LW,1.6)
title('Inverse of the logit-normal distribution',FS,12)
```

![ResamplingRandomVariables figure 05](../../images/stats/ResamplingRandomVariables_05.png)

To apply the result for resampling, we have to reflect uniform values less than $1/2$ back into $[1/2,1]$, and reflect the results back.

```matlab
u = rand(1e4,1);
flag = (u < 0.5);  u(flag) = 1-u(flag);
x = cdfinv( u );  x(flag) = 1-x(flag);
[count,bin] = hist(x,36);
count = count/sum(count*(bin(2)-bin(1)));  % renormalize, total area = 1
clf, bar(bin,count), hold on
plot(density,'r',LW,1.6), axis tight
title('Sampled points and the orignal density',FS,12)
```

![ResamplingRandomVariables figure 06](../../images/stats/ResamplingRandomVariables_06.png)

We can see what our truncation of the original random variable costs us by looking at the domain of the inverse cdf:

```matlab
cdfinv.ends.'
missing = 1 - ans(end)
```

```text
missing =
     2.449569436180354e-10
```

Thus, a uniform variable that takes a value closer to 1 than this number won't be mapped accurately back to the logit-normal variable we want, unless we take further steps. Clearly, such events will be extremely rare.

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
