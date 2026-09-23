# Advection-diffusion in the unit ball

*Nicolas Boullé, July 2019*

[Original MATLAB Chebfun example](https://www.chebfun.org/examples/sphere/AdvectionDiffusion.html)

Python translation: [`examples/sphere/advectiondiffusion.py`](https://github.com/ma-gilles/chebfunjax/blob/main/examples/sphere/advectiondiffusion.py)

## Introduction

In this example, we use the Helmholtz solver of Ballfun, available through the `helmholtz` command, to solve the advection-diffusion equation in the unit ball. We also use some of the vector calculus and visualization capabilities of Ballfun.

## Advection-diffusion in the ball

The advection-diffusion equation in the ball is given by $$ \frac{\partial c}{\partial t}=D\nabla^2c-v\cdot\nabla c, $$ where $D$ is the diffusion coefficient and $v$ is a divergence-free vector field.

We choose $D=1/5000$ and $v = \nabla\times[ze^{-5(x^2+y^2+z^2)}(x,y,z)]$ to satisfy the no-slip condition $v\cdot \vec{n}=0$.

```matlab
w = ballfunv( @(x,y,z) z.*exp(-5*(x.^2+y.^2+z.^2)).*x,...
              @(x,y,z) z.*exp(-5*(x.^2+y.^2+z.^2)).*y,...
              @(x,y,z) z.*exp(-5*(x.^2+y.^2+z.^2)).*z);
v = curl( w );
quiver(v, 4, 'numpts',30), axis('off'), colorbar
```

![AdvectionDiffusion figure 01](../../images/sphere/AdvectionDiffusion_01.png)

We verify that $v$ is divergence-free:

```matlab
norm(div(v))
```

```text
ans =
     5.4268e-30
```

The vector field $v$ also satisfies the no-slip boundary condition $v\cdot\vec{n}=0$, as shown by the following command:

```matlab
vn = dot(v(1,:,:,'spherical'),spherefunv.unormal);
norm(vn)
```

```text
ans =
     2.2839e-18
t=0 plotted (0s)
t=5 plotted (30s)
t=10 plotted (47s)
t=15 plotted (65s)
done (65s)
```

## Vizualisation of functions

We impose the initial condition $c = -xe^{-5(x^2+y^2+z^2)}$ to solve the advection-diffusion equation.

```matlab
c = ballfun(@(x,y,z) -x.*exp(-5*(x.^2+y.^2+z.^2)));
```

The function $c$ can be vizualised by the different plotting commands in Ballfun;

```matlab
subplot(2,2,1)
plot(c), caxis([-0.19 0.19]), axis off
title("Plot")

subplot(2,2,2)
slice(c), caxis([-0.19 0.19]), axis off
title("Slice")

subplot(2,2,3)
plot(c, 'WedgeAz'), caxis([-0.19 0.19]), axis off
title("WedgeAz")

subplot(2,2,4)
plot(c, 'WedgePol'), caxis([-0.19 0.19]), axis off
title("WedgePol")
```

![AdvectionDiffusion figure 02](../../images/sphere/AdvectionDiffusion_02.png)

## Time discretization

Now we solve the advection-diffusion equation numerically using the implicit-explicit order 1 backward differentiation time-stepping scheme (IMEX-BDF1). This yields a Helmholtz equation at each time step: $$ \nabla^2c^{n+1}+K^2c^{n+1}=K^2c^n+\frac{1}{D}v\cdot\nabla c^n,\quad \left.\frac{\partial c}{\partial \vec{n}}\right|_{\partial B(0,1)} = 0, $$ where $c_n$ denotes the solution at time $t = n\Delta t$, $\Delta t = 0.1$ is the time step, and $K^2 = -1/(D\Delta t)$. This equation can be solved by using the Ballfun command `helmholtz`.

The following code solves the advection-diffusion numerically to time $t=15$ and plots the solution $c$ at different times.

```matlab
D = 1/5000;                                     % Diffusion constant
dt = 0.1;                                      % Time step
K = 1i*sqrt(1/(dt*D));                          % Helmholtz frequency
T = 15;                                         % Stopping time
nsteps = ceil(T/dt);                            % Number of time steps
m = 100;                                        % Spatial discretization

for n = 0:nsteps
    if mod(n,50) == 0
        clf, slice(c), caxis([-0.2,0.2])
        title(sprintf('Time %d',n*dt)), colorbar, axis('off'), snapnow
    end

    rhs = K^2*c+dot(v,grad(c))/D;
    c = helmholtz(rhs, K, @(x,y,z)0, m, 'neumann');
end
```

![AdvectionDiffusion figure 03](../../images/sphere/AdvectionDiffusion_03.png)

![AdvectionDiffusion figure 04](../../images/sphere/AdvectionDiffusion_04.png)

![AdvectionDiffusion figure 05](../../images/sphere/AdvectionDiffusion_05.png)

*(Figure 06 of the original page is not reproduced yet.)*

---

*Translated with [chebfunjax](https://github.com/ma-gilles/chebfunjax); prose and MATLAB code from the original example, copyright The University of Oxford and The Chebfun Developers.  Printed outputs and figures are chebfunjax's.*
