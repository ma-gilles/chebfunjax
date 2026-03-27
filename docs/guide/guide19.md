# Chapter 19: SPIN, SPIN2, SPIN3 and SPINSPHERE for Stiff PDEs

*Based on [Chebfun Guide Chapter 19](https://www.chebfun.org/docs/guide/guide19.html)*

The `spin` module in chebfunjax provides exponential time-differencing integrators for stiff semilinear PDEs in 1D, 2D, 3D, and on the sphere. The name "SPIN" stands for "Stiff PDE INtegrator."

## 19.1 Introduction

Many important PDEs have the form

$$u_t = S(u) = Lu + N(u)$$

where $L$ is a constant-coefficient linear differential operator and $N$ is a nonlinear operator. Examples include:

| PDE | Equation | Stiff term |
|-----|----------|------------|
| Allen-Cahn | $u_t = \epsilon u_{xx} + u - u^3$ | $\epsilon u_{xx}$ |
| KdV | $u_t = -u_{xxx} - \tfrac{1}{2}(u^2)_x$ | $-u_{xxx}$ |
| NLS | $u_t = i u_{xx} + i\|u\|^2 u$ | $iu_{xx}$ |
| Kuramoto-Sivashinsky | $u_t = -u_{xx} - u_{xxxx} - \tfrac{1}{2}(u^2)_x$ | $-u_{xxxx}$ |

These equations are stiff because $L$ has eigenvalues that scale as high powers of the wavenumber $k$. The `spin` solvers combine Fourier spectral discretization in space with the ETDRK4 (exponential time-differencing Runge-Kutta, order 4) scheme in time.

The four solver functions are:

| Solver | Domain | Module |
|--------|--------|--------|
| `spin` | 1D periodic interval | `chebfunjax.spin` |
| `spin2` | 2D periodic rectangle | `chebfunjax.spin` |
| `spin3` | 3D periodic cuboid | `chebfunjax.spin` |
| `spinsphere` | Unit sphere | `chebfunjax.spin` |

## 19.2 Computations in 1D with spin

### Built-in examples

The simplest way to use `spin` is with a built-in PDE name:

```python
from chebfunjax.spin import spin, SpinOp

# Solve the KdV equation (two-soliton interaction)
x, t, u = spin('KdV', N=512, dt=3e-6)
print(f"Final time: {t}")
print(f"Grid size: {len(x)}")
```

Available built-in 1D PDEs:

| Name | PDE | Domain |
|------|-----|--------|
| `'AC'` | Allen-Cahn: $u_t = 5 \times 10^{-3} u_{xx} + u - u^3$ | $[0, 2\pi]$ |
| `'KdV'` | Korteweg-de Vries: $u_t = -u_{xxx} - \tfrac{1}{2}(u^2)_x$ | $[-\pi, \pi]$ |
| `'NLS'` | Nonlinear Schrodinger: $u_t = iu_{xx} + i\|u\|^2 u$ | $[-\pi, \pi]$ |
| `'KS'` | Kuramoto-Sivashinsky: $u_t = -u_{xx} - u_{xxxx} - \tfrac{1}{2}(u^2)_x$ | $[0, 32\pi]$ |

### Return values

`spin` returns:
- `x`: spatial grid points, shape `(N,)`
- `t`: final time reached (float)
- `u_final`: solution values at the final time, shape `(N,)`

```python
x, t, u = spin('AC', N=256, dt=0.1)

# Plot the result (using matplotlib)
import matplotlib.pyplot as plt
plt.plot(x, u)
plt.title(f'Allen-Cahn at t = {t}')
plt.show()
```

### Custom SpinOp

For PDEs not in the built-in catalogue, create a `SpinOp` directly:

```python
import jax.numpy as jnp

# Viscous Burgers equation: u_t = nu*u_xx - 0.5*(u^2)_x
nu = 1e-3
op = SpinOp(
    lin_coeff=lambda k: nu * (1j * k)**2,       # nu * d^2/dx^2
    nonlin_vals=lambda u: -0.5 * u**2,           # -0.5 * u^2
    nonlin_diff_order=1,                          # differentiate once
    domain=(-1.0, 1.0),
    tspan=(0.0, 20.0),
    u0=lambda x: (1 - x**2) * jnp.exp(-30 * (x + 0.5)**2),
    is_real=True,
)

x, t, u = spin(op, N=512, dt=5e-3)
```

The key parameters of `SpinOp`:

- **`lin_coeff`**: A function of the wavenumber array `k` that returns the diagonal of the linear operator in Fourier space. For example, $\nu \partial^2/\partial x^2$ has eigenvalues $\nu (ik)^2 = -\nu k^2$.

- **`nonlin_vals`**: A function that evaluates the nonlinear part in physical (value) space. For Burgers, the nonlinearity is $-\tfrac{1}{2} u^2$ (before differentiation).

- **`nonlin_diff_order`**: The number of spatial derivatives to apply to the nonlinear term in Fourier space. For $-\tfrac{1}{2}(u^2)_x$, set this to 1 so that the Fourier coefficients are multiplied by $(ik)$.

- **`is_real`**: Set `True` for real-valued PDEs (imaginary parts are discarded).

### Dealiasing

By default, the 2/3-rule dealiasing is applied at each step (zeroing the top third of Fourier modes). This can be disabled:

```python
x, t, u = spin('KdV', N=512, dt=3e-6, dealias=False)
```

### Verbose output

Set `verbose=True` to print progress every 10% of the integration:

```python
x, t, u = spin('KS', N=256, dt=1e-2, verbose=True)
```

## 19.3 The ETDRK4 Scheme

The solver uses the ETDRK4 scheme (Cox and Matthews, 2002; Kassam and Trefethen, 2005). In Fourier space, the PDE becomes the system of ODEs

$$\hat{u}_t = L \hat{u} + \hat{N}(u)$$

where $L$ is diagonal (the Fourier eigenvalues of the linear operator) and $\hat{N}$ is the Fourier transform of the nonlinear term.

The four ETDRK4 stages are:

$$a = e^{\Delta t/2 \cdot L}\, \hat{u}_n$$
$$b = e^{\Delta t/2 \cdot L}\, \hat{u}_n + \psi_{1,2}\, \hat{N}(a)$$
$$c = e^{\Delta t \cdot L}\, \hat{u}_n + 2\psi_{1,2}\, \hat{N}(b)$$
$$\hat{u}_{n+1} = e^{\Delta t \cdot L}\, \hat{u}_n + B_2\, \hat{N}(a) + B_3\, \hat{N}(b) + B_4\, \hat{N}(c)$$

where $\psi_{1,2} = \frac{\Delta t}{2}\, \varphi_1(\frac{\Delta t}{2} L)$ and the $B$ coefficients involve the $\varphi$-functions $\varphi_2$ and $\varphi_3$.

The $\varphi$-functions are evaluated stably via contour integration on small circles of radius 1 in the complex plane (Kassam and Trefethen, 2005), avoiding the cancellation issues of direct formulas.

## 19.4 Computations in 2D with spin2

The `spin2` function solves 2D periodic PDEs on rectangles:

```python
from chebfunjax.spin import spin2, SpinOp2

# Ginzburg-Landau equation in 2D
# u_t = Delta(u) + u - (1 + 1.5i)*u*|u|^2
xx, yy, t, u = spin2('GL', N=64, dt=5e-3)
print(f"Grid shape: {xx.shape}")
print(f"Final time: {t}")
```

Available built-in 2D PDEs:

| Name | PDE |
|------|-----|
| `'AC2'` | Allen-Cahn 2D: $u_t = 5 \times 10^{-3} \Delta u + u - u^3$ |
| `'GL'` | Ginzburg-Landau: $u_t = \Delta u + u - (1+1.5i)u\|u\|^2$ |
| `'GS'` | Gray-Scott (2-component): reaction-diffusion stripes |
| `'SH'` | Swift-Hohenberg: $u_t = -2\Delta u - \Delta^2 u - 0.9u - u^3$ |

### Custom SpinOp2

For 2D PDEs, the linear operator must be an isotropic polynomial in the Laplacian:

$$L = A\, \Delta + B\, \Delta^2 + C\, \Delta^3 + D\, \Delta^4 + E\, \Delta^5$$

```python
# Swift-Hohenberg: L = -2*Delta - Delta^2
# N(u) = -0.9*u - u^3
op = SpinOp2(
    lin_coeffs=(-2.0, -1.0, 0.0, 0.0, 0.0),  # (A, B, C, D, E)
    nonlin_vals=lambda u: -0.9 * u - u**3,
    n_vars=1,
    domain=(0.0, 20.0, 0.0, 20.0),
    tspan=(0.0, 100.0),
    u0=lambda x, y: jnp.cos(x / 16) * jnp.sin(y / 16),
    is_real=True,
)

xx, yy, t, u = spin2(op, N=64, dt=1e-2)
```

### Multi-component PDEs

The Gray-Scott equations have two components. For multi-component PDEs, pass lists of `lin_coeffs`, `nonlin_vals`, and `u0`:

```python
op = SpinOp2.from_name('GS')
xx, yy, t, u = spin2(op, N=64, dt=1e-1)
# u is a list of two (N, N) arrays: [u_component, v_component]
```

## 19.5 Computations in 3D with spin3

The `spin3` function solves 3D periodic PDEs:

```python
from chebfunjax.spin import spin3, SpinOp3

# 3D Ginzburg-Landau
grids, t, u = spin3('GL', N=32, dt=1e-1)
xx, yy, zz = grids
print(f"Grid shape: {xx.shape}")
```

Available built-in 3D PDEs:

| Name | PDE |
|------|-----|
| `'GL'` | Ginzburg-Landau: $u_t = \Delta u + u - (1+1.5i)u\|u\|^2$ |
| `'SH'` | Swift-Hohenberg: $u_t = -2\Delta u - \Delta^2 u - 0.9u - u^3$ |
| `'AC'` | Allen-Cahn: $u_t = \epsilon \Delta u + u - u^3$ |

### Custom SpinOp3

```python
# Allen-Cahn in 3D with custom parameters
op = SpinOp3(
    lin_scales=(5e-3,),
    lin_ops=('lap',),
    nonlin_vals=lambda u: u - u**3,
    domain=(0.0, 2*jnp.pi, 0.0, 2*jnp.pi, 0.0, 2*jnp.pi),
    tspan=(0.0, 100.0),
    u0=lambda x, y, z: jnp.tanh(jnp.sin(x) * jnp.cos(y) * jnp.sin(z)),
    is_real=True,
)

grids, t, u = spin3(op, N=32, dt=5e-2)
```

## 19.6 Computations on the Sphere with spinsphere

The `spinsphere` function solves PDEs on the unit sphere using the doubled Fourier sphere (DFS) method:

```python
from chebfunjax.spin import spinsphere, SpinOpSphere

# Allen-Cahn on the sphere
grids, t, u = spinsphere('AC', N=32, dt=5e-3)
ll, tt = grids  # longitude, doubled colatitude
print(f"Grid shape: {ll.shape}")
print(f"Final time: {t}")
```

Available built-in sphere PDEs:

| Name | PDE |
|------|-----|
| `'AC'` | Allen-Cahn: $u_t = 10^{-2}\, \Delta_S u + u - u^3$ |
| `'GL'` | Ginzburg-Landau: $u_t = 10^{-3}\, \Delta_S u + u - (1+1.5i)u\|u\|^2$ |
| `'NLS'` | Nonlinear Schrodinger: $u_t = i\, \Delta_S u + i\|u\|^2 u$ |

Here $\Delta_S$ is the Laplace-Beltrami operator on the sphere.

### Custom SpinOpSphere

```python
# Diffusion-reaction on the sphere
op = SpinOpSphere(
    lin_scale=0.01,  # coefficient of the Laplace-Beltrami operator
    nonlin_vals=lambda u: u - u**3,
    tspan=(0.0, 50.0),
    u0=lambda lam, th: jnp.cos(3 * lam) * jnp.sin(th)**3,
    is_real=True,
)

grids, t, u = spinsphere(op, N=64, dt=1e-2)
```

The Laplace-Beltrami operator on the sphere in the DFS representation is NOT diagonal in Fourier space (it is block-tridiagonal). The `spinsphere` solver handles this by computing matrix exponentials block-by-block.

## 19.7 Solver Options

All four solvers accept common keyword arguments:

- **`N`**: Number of Fourier modes (per direction). Larger $N$ gives better spatial resolution.
- **`dt`**: Time-step. Smaller $dt$ gives better temporal accuracy but costs more.
- **`dealias`**: Apply 2/3-rule dealiasing (default `True`).
- **`M`**: Number of contour points for $\varphi$-function evaluation (default 32).
- **`verbose`**: Print progress (default `False`).

## 19.8 A Note on History

Exponential integrators for ODEs trace back to Hersch (1958) and Certaine (1960). Cox and Matthews (2002) introduced the ETDRK4 scheme for stiff PDEs. Kassam and Trefethen (2005) improved its numerical stability using contour integral evaluation of the $\varphi$-functions. Montanelli and Bootland (2017) extended the approach to 2D, 3D, and spherical domains and implemented the MATLAB Chebfun `spin` package.

## 19.9 References

1. S. M. Cox and P. C. Matthews, "Exponential time differencing for stiff systems", *J. Comput. Phys.*, 176, 430--455, 2002.

2. A.-K. Kassam and L. N. Trefethen, "Fourth-order time-stepping for stiff PDEs", *SIAM J. Sci. Comput.*, 26(4), 1214--1233, 2005.

3. H. Montanelli and N. Bootland, "Solving periodic semilinear stiff PDEs in 1D, 2D and 3D with exponential integrators", *Math. Comp.*, 89, 1493--1524, 2020.

4. M. Hochbruck and A. Ostermann, "Exponential integrators", *Acta Numer.*, 19, 209--286, 2010.
