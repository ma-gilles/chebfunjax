# Geometric Brownian motion

*Nick Trefethen, May 2017*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/ode-random/GBM.html)

Python translation: [`examples/ode-random/gbm.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/ode-random/gbm.py)

Random ODEs and stochastic DEs may include *additive noise* and/or *multiplicative noise*. A linear, constant-coefficient equation of the latter kind is the equation of *geometric Brownian motion*, $$ dX_t = \mu X_t dt + \sigma X_t dW_t, ~~~ (1) $$ where $W_t$ is the Wiener process (Brownian motion). With Chebfun's smooth random functions the analogous equation is $$ y' = \mu y + \sigma f y, ~~~ (2) $$ where $f$ is a smooth random function. As usual, $f$ will have a wavelength parameter $\lambda>0$, and the SDE limit corresponds to $\lambda \to 0$. Actually in this limit one will get a Stratonovich (rather than Itô) SDE, written $$ dX_t = \mu X_t dt + \sigma X_t \circ dW_t. ~~~ (3) $$

$\mu$ is called the *drift* coefficient and $\sigma$ is the *diffusion* (or sometimes *volatility*) coefficient.

Geometric Brownian motion is easy to analyze by taking the logarithm. For example, dividing (2) by $y$ gives $$ (\log y)' = \mu + \sigma f , $$ which now involves just additive noise.

For example, here are five trajectories with $\mu = 0$ and $\sigma = 1$. On a log scale there would be no bias up or down, but on a linear scale we see some large amplitudes.

```matlab
tic
dom = [0,20]; L = chebop(dom); L.lbc = 1; L.maxnorm = 100;
rng(0), lambda = 0.2;
f = randnfun(lambda,dom,'big',5);
mu = 0; sigma = 1;
for k = 1:5
   L.op = @(t,y) diff(y) - mu*y - sigma*f(:,k)*y;
   y = L\0; plot(y), hold on
end
grid on, hold off
xlabel('t'), ylabel('y')
title('zero drift')
```

![GBM figure 01](../../images/ode-random/GBM_01.png)

If we increase $\mu$ to $0.2$, there is now an upward bias on any scale.

```matlab
mu = 0.2;
for k = 1:5
   L.op = @(t,y) diff(y) - mu*y - sigma*f(:,k)*y;
   y = L\0; plot(y), hold on
end
grid on, hold off, ylim([0 70])
xlabel('t'), ylabel('y')
title('positive drift')
```

![GBM figure 02](../../images/ode-random/GBM_02.png)

Setting $\mu = -0.2$, on the other hand, leads to decay.

```matlab
mu = -0.2;
for k = 1:5
   L.op = @(t,y) diff(y) - mu*y - sigma*f(:,k)*y;
   y = L\0; plot(y), hold on
end
grid on, hold off
xlabel('t'), ylabel('y')
title('negative drift')
```

![GBM figure 03](../../images/ode-random/GBM_03.png)

```matlab
total_time_in_seconds = toc
```

```text
total_time_in_seconds =
  728.138503
```

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
