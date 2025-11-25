# JAX Migration Quick Reference

## Quick Start

### Installation
```bash
# CPU-only
pip install jax

# GPU (CUDA 12.x)
pip install jax[cuda12]
```

### Basic Imports
```python
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
```

## Common Patterns Cheat Sheet

### Array Creation and Manipulation

| Operation | NumPy | JAX |
|-----------|-------|-----|
| Create array | `np.array([1,2,3])` | `jnp.array([1,2,3])` |
| Zeros | `np.zeros((3,3))` | `jnp.zeros((3,3))` |
| Ones | `np.ones((3,3))` | `jnp.ones((3,3))` |
| Random | `np.random.rand(3,3)` | `jax.random.normal(key, (3,3))` |
| Reshape | `arr.reshape(...)` | `arr.reshape(...)` |
| Transpose | `arr.T` or `np.transpose(arr)` | `arr.T` or `jnp.transpose(arr)` |

### Array Updates (Most Important!)

| Operation | NumPy (In-place) | JAX (Functional) |
|-----------|------------------|------------------|
| Set value | `arr[i] = val` | `arr = arr.at[i].set(val)` |
| Add to value | `arr[i] += val` | `arr = arr.at[i].add(val)` |
| Multiply value | `arr[i] *= val` | `arr = arr.at[i].multiply(val)` |
| Set slice | `arr[1:3] = vals` | `arr = arr.at[1:3].set(vals)` |
| Set 2D | `arr[i,j] = val` | `arr = arr.at[i,j].set(val)` |

### Mathematical Operations

| Operation | NumPy | JAX |
|-----------|-------|-----|
| Sum | `np.sum(arr)` | `jnp.sum(arr)` |
| Mean | `np.mean(arr)` | `jnp.mean(arr)` |
| Std | `np.std(arr)` | `jnp.std(arr)` |
| Min/Max | `np.min(arr)`, `np.max(arr)` | `jnp.min(arr)`, `jnp.max(arr)` |
| Argmin/Argmax | `np.argmin(arr)`, `np.argmax(arr)` | `jnp.argmin(arr)`, `jnp.argmax(arr)` |
| Dot product | `np.dot(a, b)` | `jnp.dot(a, b)` |
| Matrix multiply | `a @ b` | `a @ b` |

### FFT Operations

| Operation | NumPy/CuPy | JAX |
|-----------|------------|-----|
| 1D FFT | `np.fft.fft(x)` | `jnp.fft.fft(x)` |
| 2D FFT | `np.fft.fft2(x)` | `jnp.fft.fft2(x)` |
| Real FFT | `np.fft.rfft(x)` | `jnp.fft.rfft(x)` |
| 2D Real FFT | `np.fft.rfft2(x)` | `jnp.fft.rfft2(x)` |
| Inverse FFT | `np.fft.ifft(x)` | `jnp.fft.ifft(x)` |
| Inverse Real FFT | `np.fft.irfft(x)` | `jnp.fft.irfft(x)` |

### Conditional Operations

| Operation | NumPy | JAX |
|-----------|-------|-----|
| Where | `np.where(cond, x, y)` | `jnp.where(cond, x, y)` |
| Clip | `np.clip(arr, min, max)` | `jnp.clip(arr, min, max)` |
| Maximum | `np.maximum(a, b)` | `jnp.maximum(a, b)` |
| Minimum | `np.minimum(a, b)` | `jnp.minimum(a, b)` |

### Random Number Generation

| Operation | NumPy | JAX |
|-----------|-------|-----|
| Uniform | `np.random.rand(n)` | `jax.random.uniform(key, (n,))` |
| Normal | `np.random.randn(n)` | `jax.random.normal(key, (n,))` |
| Choice | `np.random.choice(arr, n)` | `jax.random.choice(key, arr, (n,))` |
| Permutation | `np.random.permutation(arr)` | `jax.random.permutation(key, arr)` |

**Important:** JAX requires explicit key management:
```python
key = jax.random.PRNGKey(0)  # Initialize
key, subkey = jax.random.split(key)  # Split for each use
random_data = jax.random.normal(subkey, (100,))
```

## JIT Compilation

### Basic JIT
```python
# Before (NumPy)
def slow_func(x):
    return np.sum(x ** 2)

# After (JAX)
@jax.jit
def fast_func(x):
    return jnp.sum(x ** 2)
```

### JIT with Static Arguments
```python
# If some arguments should not trigger recompilation
import functools

@functools.partial(jax.jit, static_argnums=(1, 2))
def func(x, shape, dtype):
    return jnp.zeros(shape, dtype=dtype) + x
```

### When to Use JIT
- ✅ Functions called many times
- ✅ Functions with heavy computation
- ✅ Functions with static structure
- ❌ Functions with dynamic shapes
- ❌ Functions with Python control flow

## Vectorization (vmap)

### Basic vmap
```python
# Process single item
def process_one(x):
    return x ** 2 + 1

# Process batch
process_batch = jax.vmap(process_one)

# Usage
single = jnp.array(5.0)
batch = jnp.array([1.0, 2.0, 3.0, 4.0])

result_single = process_one(single)  # scalar
result_batch = process_batch(batch)  # array
```

### vmap with Multiple Arguments
```python
def compute_distance(p1, p2):
    return jnp.sqrt(jnp.sum((p1 - p2) ** 2))

# Vectorize over first argument
compute_distances = jax.vmap(compute_distance, in_axes=(0, None))

# Usage
points = jnp.array([[1,2], [3,4], [5,6]])
reference = jnp.array([0,0])
distances = compute_distances(points, reference)
```

## Control Flow

### Simple Conditionals
```python
# Use jnp.where for simple conditions
result = jnp.where(x > 0, x, 0)  # ReLU
```

### Complex Conditionals
```python
# Use jax.lax.cond for complex conditions
def true_func():
    return x ** 2

def false_func():
    return x + 1

result = jax.lax.cond(x > 0, true_func, false_func)
```

### Loops with scan
```python
# Cumulative sum using scan
def cumsum(carry, x):
    new_carry = carry + x
    return new_carry, new_carry

init_carry = 0
final_carry, outputs = jax.lax.scan(cumsum, init_carry, jnp.array([1,2,3,4,5]))
# outputs = [1, 3, 6, 10, 15]
```

## Device Management

### Check Available Devices
```python
import jax

print(jax.devices())  # List all devices
print(jax.default_backend())  # Default backend (cpu/gpu/tpu)
```

### Explicit Device Placement
```python
# Put array on CPU
cpu_device = jax.devices('cpu')[0]
x_cpu = jax.device_put(x, cpu_device)

# Put array on GPU
gpu_devices = jax.devices('gpu')
if gpu_devices:
    x_gpu = jax.device_put(x, gpu_devices[0])
```

### Multi-Device Parallelism
```python
# Replicate computation across devices
@jax.pmap
def parallel_square(x):
    return x ** 2

# Input shape: (n_devices, ...)
n_devices = jax.local_device_count()
x = jnp.arange(n_devices * 100).reshape(n_devices, 100)
result = parallel_square(x)
```

## Common Migration Patterns

### Pattern: Numba → JAX

```python
# Before (Numba)
from numba import jit

@jit(nopython=True)
def numba_func(x, y):
    return x + y

# After (JAX)
@jax.jit
def jax_func(x, y):
    return x + y
```

### Pattern: CuPy → JAX

```python
# Before (CuPy)
import cupy as cp

x_gpu = cp.asarray(x)
result = cp.sum(x_gpu ** 2)
result_cpu = result.get()

# After (JAX)
import jax.numpy as jnp

x = jnp.asarray(x)  # Automatic device placement
result = jnp.sum(x ** 2)  # Stays on device
result_cpu = np.array(result)  # Transfer to CPU if needed
```

### Pattern: Parallel Loop → vmap

```python
# Before (Numba)
from numba import jit, prange

@jit(nopython=True, parallel=True)
def parallel_process(arr):
    result = np.empty(len(arr))
    for i in prange(len(arr)):
        result[i] = arr[i] ** 2
    return result

# After (JAX)
@jax.jit
def parallel_process(arr):
    return jax.vmap(lambda x: x ** 2)(arr)

# Or simply
@jax.jit
def parallel_process(arr):
    return arr ** 2  # Automatically vectorized
```

## Debugging Tips

### Print Values in JIT Functions
```python
# Use jax.debug.print (not regular print)
@jax.jit
def debug_func(x):
    jax.debug.print("x = {}", x)
    return x ** 2
```

### Disable JIT for Debugging
```python
# Temporarily disable JIT
with jax.disable_jit():
    result = jitted_function(x)
```

### Check for NaNs
```python
# Check for NaN/Inf values
jax.config.update("jax_debug_nans", True)
```

## Performance Tips

### 1. Minimize Data Transfer
```python
# Bad - multiple transfers
x_jax = jnp.array(x_numpy)
y_jax = jnp.array(y_numpy)
result = jnp.dot(x_jax, y_jax)

# Good - batch transfer
x_numpy, y_numpy = ...
x_jax, y_jax = jnp.array([x_numpy, y_numpy])
result = jnp.dot(x_jax, y_jax)
```

### 2. Use Static Arguments
```python
# Avoid recompilation by marking static args
@functools.partial(jax.jit, static_argnums=(1, 2))
def create_array(x, shape, dtype):
    return jnp.zeros(shape, dtype=dtype) + x
```

### 3. Batch Operations
```python
# Bad - loop in Python
results = []
for x in inputs:
    results.append(process(x))

# Good - vectorize with vmap
results = jax.vmap(process)(inputs)
```

### 4. Avoid Python Loops in JIT
```python
# Bad
@jax.jit
def bad_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

# Good
@jax.jit
def good_sum(arr):
    return jnp.sum(arr)
```

## Testing Numerical Equivalence

```python
import numpy as np
import jax.numpy as jnp

def test_equivalence():
    # NumPy version
    x_np = np.random.rand(100)
    result_np = numpy_implementation(x_np)
    
    # JAX version
    x_jax = jnp.array(x_np)
    result_jax = jax_implementation(x_jax)
    
    # Compare
    np.testing.assert_allclose(
        result_np, 
        np.array(result_jax),
        rtol=1e-6,
        atol=1e-6
    )
```

## Common Errors and Solutions

### Error: "Array is immutable"
```python
# Problem
arr[0] = 5  # TypeError

# Solution
arr = arr.at[0].set(5)
```

### Error: "Tracer values cannot be used in Python control flow"
```python
# Problem
@jax.jit
def bad_func(x):
    if x > 0:  # Error!
        return x
    return -x

# Solution
@jax.jit
def good_func(x):
    return jnp.where(x > 0, x, -x)
```

### Error: "Random key reuse"
```python
# Problem
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
y = jax.random.normal(key, (10,))  # Same as x!

# Solution
key = jax.random.PRNGKey(0)
key, subkey1 = jax.random.split(key)
x = jax.random.normal(subkey1, (10,))
key, subkey2 = jax.random.split(key)
y = jax.random.normal(subkey2, (10,))  # Different from x
```

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub](https://github.com/google/jax)
- [Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)

## Quick Commands

```bash
# Check JAX version
python -c "import jax; print(jax.__version__)"

# List devices
python -c "import jax; print(jax.devices())"

# Run tests
pytest tests/ -v

# Profile code
python -m cProfile -s cumtime script.py
```
