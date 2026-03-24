# Installation

## Requirements

- Python >= 3.10
- JAX >= 0.4.20
- NumPy >= 1.24, SciPy >= 1.11
- [Equinox](https://docs.kidger.site/equinox/) >= 0.11

---

## pip (CPU)

```bash
git clone https://github.com/ma-gilles/chebfunjax.git
cd chebfunjax
pip install -e .
```

Verify the install:

```python
import chebfunjax as cj
print(cj.__version__)   # 0.1.0
```

---

## pixi (recommended — handles JAX + CUDA automatically)

[pixi](https://pixi.sh) is a fast conda-based environment manager that resolves
JAX, CUDA, and all transitive dependencies in one step.

```bash
git clone https://github.com/ma-gilles/chebfunjax.git
cd chebfunjax
pixi install
pixi run smoke   # prints "chebfunjax imported OK" if everything works
```

---

## GPU Setup (CUDA)

To use a GPU, install JAX with CUDA support.

### With pixi (automatic)

The `pixi.toml` already includes the CUDA variant of JAX.
After `pixi install`, verify that JAX sees your GPU:

```bash
pixi run python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0)]
```

### With pip (manual)

Follow the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html)
for your CUDA version, e.g.:

```bash
pip install -U "jax[cuda12]"
```

Then install chebfunjax normally:

```bash
pip install -e .
```

---

## float64

chebfunjax **always** enables float64 at import time:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

This happens automatically when you `import chebfunjax`.
Spectral methods need double precision to achieve 1e-12 accuracy; running in
float32 would silently give wrong results.

---

## Development Install

```bash
pip install -e ".[dev]"   # includes ruff, pytest, pytest-xdist, pytest-cov
```

Run the test suite (CPU, non-slow tests only):

```bash
pytest -m "not slow and not gpu and not matlab"
```
