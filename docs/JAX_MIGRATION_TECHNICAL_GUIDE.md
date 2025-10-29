# JAX Migration Technical Guide

## Overview

This guide provides detailed technical information for developers migrating BM3D-ORNL code from NumPy/Numba/CuPy to JAX. It includes concrete examples, common patterns, and troubleshooting tips.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Core Concepts](#core-concepts)
3. [Migration Patterns](#migration-patterns)
4. [Module-Specific Guides](#module-specific-guides)
5. [Testing Strategies](#testing-strategies)
6. [Performance Optimization](#performance-optimization)
7. [Common Pitfalls](#common-pitfalls)

## Environment Setup

### Installation

```bash
# CPU-only installation
pip install jax jaxlib

# GPU installation (CUDA 12.x)
pip install jax[cuda12]

# Or using conda
conda install -c conda-forge jax
```

### Verification

```python
import jax
import jax.numpy as jnp

# Check available devices
print(f"Available devices: {jax.devices()}")

# Test GPU availability
print(f"Default backend: {jax.default_backend()}")

# Simple computation
x = jnp.array([1, 2, 3])
y = jnp.array([4, 5, 6])
print(f"Dot product: {jnp.dot(x, y)}")
```

## Core Concepts

### JAX Arrays vs NumPy Arrays

```python
import numpy as np
import jax.numpy as jnp

# NumPy - mutable
np_array = np.array([1, 2, 3])
np_array[0] = 10  # This works

# JAX - immutable
jax_array = jnp.array([1, 2, 3])
# jax_array[0] = 10  # This raises an error!

# Correct way in JAX
jax_array = jax_array.at[0].set(10)
```

### JIT Compilation

```python
import jax
import jax.numpy as jnp

# Without JIT
def slow_function(x):
    return jnp.sum(x ** 2)

# With JIT
@jax.jit
def fast_function(x):
    return jnp.sum(x ** 2)

# First call compiles the function
x = jnp.arange(1000000)
result = fast_function(x)  # Compilation + execution

# Subsequent calls are much faster
result = fast_function(x)  # Just execution
```

### Automatic Vectorization

```python
import jax
import jax.numpy as jnp

# Process a single item
def process_item(x):
    return x ** 2 + 2 * x + 1

# Process a batch using vmap
process_batch = jax.vmap(process_item)

# Usage
single_item = jnp.array(5.0)
batch = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

result_single = process_item(single_item)
result_batch = process_batch(batch)
```

## Migration Patterns

### Pattern 1: NumPy to JAX Array Operations

```python
# Before (NumPy)
import numpy as np

def normalize_array(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std

# After (JAX)
import jax.numpy as jnp

@jax.jit
def normalize_array(arr):
    mean = jnp.mean(arr)
    std = jnp.std(arr)
    return (arr - mean) / std
```

### Pattern 2: In-Place Operations

```python
# Before (NumPy) - in-place modification
import numpy as np

def update_array(arr, indices, values):
    arr[indices] = values
    return arr

# After (JAX) - functional update
import jax.numpy as jnp

@jax.jit
def update_array(arr, indices, values):
    return arr.at[indices].set(values)

# Or for accumulation
def accumulate_array(arr, indices, values):
    return arr.at[indices].add(values)
```

### Pattern 3: Numba JIT to JAX JIT

```python
# Before (Numba)
from numba import jit

@jit(nopython=True)
def compute_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# After (JAX)
import jax
import jax.numpy as jnp

@jax.jit
def compute_distance(p1, p2):
    return jnp.sqrt(jnp.sum((p1 - p2) ** 2))
```

### Pattern 4: Numba Parallel Loops to JAX vmap

```python
# Before (Numba with prange)
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def parallel_process(inputs):
    results = np.empty(len(inputs))
    for i in prange(len(inputs)):
        results[i] = inputs[i] ** 2
    return results

# After (JAX with vmap)
import jax
import jax.numpy as jnp

@jax.jit
def process_single(x):
    return x ** 2

@jax.jit
def parallel_process(inputs):
    return jax.vmap(process_single)(inputs)
```

### Pattern 5: CuPy to JAX

```python
# Before (CuPy)
import cupy as cp

def gpu_fft(signal):
    # Send to GPU
    signal_gpu = cp.asarray(signal)
    # Compute FFT
    freq_gpu = cp.fft.fft(signal_gpu)
    # Return to CPU
    return freq_gpu.get()

# After (JAX) - automatic device handling
import jax.numpy as jnp

@jax.jit
def gpu_fft(signal):
    # JAX automatically handles device placement
    return jnp.fft.fft(signal)
```

### Pattern 6: Random Number Generation

```python
# Before (NumPy)
import numpy as np

def generate_noise(shape):
    return np.random.randn(*shape)

# After (JAX)
import jax
import jax.numpy as jnp

def generate_noise(key, shape):
    return jax.random.normal(key, shape)

# Usage
key = jax.random.PRNGKey(0)
noise = generate_noise(key, (100, 100))

# For multiple calls, split the key
key, subkey = jax.random.split(key)
more_noise = generate_noise(subkey, (100, 100))
```

### Pattern 7: Conditional Operations

```python
# Before (NumPy)
import numpy as np

def clip_array(arr, threshold):
    result = arr.copy()
    result[arr > threshold] = threshold
    return result

# After (JAX)
import jax.numpy as jnp

@jax.jit
def clip_array(arr, threshold):
    return jnp.where(arr > threshold, threshold, arr)

# Or using clip
@jax.jit
def clip_array_v2(arr, threshold):
    return jnp.clip(arr, None, threshold)
```

## Module-Specific Guides

### gpu_utils.py Migration

#### hard_thresholding Function

```python
# Before (CuPy)
import cupy as cp

def hard_thresholding(hyper_block, threshold):
    hyper_block = cp.asarray(hyper_block)
    hyper_block = cp.fft.rfft2(hyper_block, axes=(1, 2, 3))
    threshold = cp.quantile(cp.abs(hyper_block), threshold)
    hyper_block[cp.abs(hyper_block) < threshold] = 0
    hyper_block = cp.fft.irfft2(hyper_block, axes=(1, 2, 3))
    denoised_block = hyper_block.get()
    del hyper_block
    return denoised_block

# After (JAX)
import jax
import jax.numpy as jnp

@jax.jit
def hard_thresholding(hyper_block, threshold):
    # FFT transform
    freq_block = jnp.fft.rfft2(hyper_block, axes=(1, 2, 3))
    
    # Compute threshold
    threshold_val = jnp.quantile(jnp.abs(freq_block), threshold)
    
    # Apply thresholding (functional)
    freq_block = jnp.where(jnp.abs(freq_block) < threshold_val, 0, freq_block)
    
    # Inverse FFT
    denoised_block = jnp.fft.irfft2(freq_block, axes=(1, 2, 3))
    
    return denoised_block
```

#### wiener_hadamard Function

```python
# Before (CuPy)
import cupy as cp
from cupyx.scipy.linalg import hadamard

def wiener_hadamard(hyper_block, sigma_squared):
    hyper_block = cp.asarray(hyper_block)
    n = hyper_block.shape[-1]
    H = hadamard(n)
    
    original_shape = hyper_block.shape
    if hyper_block.ndim == 4:
        hyper_block = hyper_block.reshape(-1, n, n)
    
    hyper_block = cp.einsum("ij,kjl->kil", H, hyper_block)
    hyper_block = cp.einsum("ijk,kl->ijl", hyper_block, H)
    
    local_mean = cp.mean(hyper_block, axis=0, keepdims=True)
    local_variance = cp.var(hyper_block, axis=0, keepdims=True)
    
    hyper_block = (1 - sigma_squared / (local_variance + 1e-8)) * (
        hyper_block - local_mean
    ) + local_mean
    mask = cp.broadcast_to(local_variance < sigma_squared, hyper_block.shape)
    hyper_block[mask] = 0
    
    hyper_block = cp.einsum("ij,kjl->kil", H, hyper_block)
    hyper_block = cp.einsum("ijk,kl->ijl", hyper_block, H) / (n * n)
    
    if original_shape != hyper_block.shape:
        hyper_block = hyper_block.reshape(original_shape)
    
    denoised_block = hyper_block.get()
    del hyper_block
    return denoised_block

# After (JAX)
import jax
import jax.numpy as jnp
from scipy.linalg import hadamard  # Generate matrix once

def generate_hadamard(n):
    """Generate Hadamard matrix."""
    return jnp.array(hadamard(n), dtype=jnp.float32)

@jax.jit
def wiener_hadamard(hyper_block, sigma_squared, H=None):
    """
    Wiener filtering using Hadamard transform.
    
    Note: H should be pre-computed for efficiency.
    """
    n = hyper_block.shape[-1]
    
    # If H not provided, generate it (not recommended in JIT)
    if H is None:
        # This will cause recompilation if n changes
        H = generate_hadamard(n)
    
    original_shape = hyper_block.shape
    if hyper_block.ndim == 4:
        hyper_block = hyper_block.reshape(-1, n, n)
    
    # Forward Hadamard transform
    hyper_block = jnp.einsum("ij,kjl->kil", H, hyper_block)
    hyper_block = jnp.einsum("ijk,kl->ijl", hyper_block, H)
    
    # Compute statistics
    local_mean = jnp.mean(hyper_block, axis=0, keepdims=True)
    local_variance = jnp.var(hyper_block, axis=0, keepdims=True)
    
    # Wiener filter
    wiener_factor = 1 - sigma_squared / (local_variance + 1e-8)
    hyper_block = wiener_factor * (hyper_block - local_mean) + local_mean
    
    # Apply mask
    hyper_block = jnp.where(local_variance < sigma_squared, 0, hyper_block)
    
    # Inverse Hadamard transform
    hyper_block = jnp.einsum("ij,kjl->kil", H, hyper_block)
    hyper_block = jnp.einsum("ijk,kl->ijl", hyper_block, H) / (n * n)
    
    # Reshape if needed
    if original_shape != hyper_block.shape:
        hyper_block = hyper_block.reshape(original_shape)
    
    return hyper_block
```

### utils.py Migration

#### find_candidate_patch_ids Function

```python
# Before (Numba)
from numba import jit

@jit(nopython=True)
def find_candidate_patch_ids(signal_patches, ref_index, cut_off_distance):
    num_patches = signal_patches.shape[0]
    ref_pos = signal_patches[ref_index]
    candidate_patch_ids = [ref_index]
    
    for i in range(ref_index + 1, num_patches):
        if (np.abs(signal_patches[i, 0] - ref_pos[0]) <= cut_off_distance[0] and
            np.abs(signal_patches[i, 1] - ref_pos[1]) <= cut_off_distance[1]):
            candidate_patch_ids.append(i)
    
    return candidate_patch_ids

# After (JAX)
import jax
import jax.numpy as jnp

@jax.jit
def find_candidate_patch_ids(signal_patches, ref_index, cut_off_distance):
    """
    Find candidate patch indices within spatial distance threshold.
    
    Returns array of indices instead of list.
    """
    num_patches = signal_patches.shape[0]
    ref_pos = signal_patches[ref_index]
    
    # Create array of indices
    indices = jnp.arange(ref_index, num_patches)
    
    # Compute distances
    row_dist = jnp.abs(signal_patches[indices, 0] - ref_pos[0])
    col_dist = jnp.abs(signal_patches[indices, 1] - ref_pos[1])
    
    # Find candidates
    mask = (row_dist <= cut_off_distance[0]) & (col_dist <= cut_off_distance[1])
    
    # Return indices where mask is True
    return indices[mask]
```

#### get_signal_patch_positions Function

```python
# Before (Numba) - uses dynamic list
from numba import jit

@jit(nopython=True)
def get_signal_patch_positions(image, patch_size, stride, background_threshold):
    i_height, i_width = image.shape
    p_height, p_width = patch_size
    
    signal_patches = []
    
    for r in range(0, i_height - p_height + 1, stride):
        for c in range(0, i_width - p_width + 1, stride):
            patch = image[r : r + p_height, c : c + p_width]
            patch_max = np.max(patch)
            if patch_max >= background_threshold:
                signal_patches.append((r, c))
    
    if len(signal_patches) == 0:
        raise ValueError("Couldn't find any signal patches!")
    
    return np.array(signal_patches)

# After (JAX) - pre-compute size and use masking
import jax
import jax.numpy as jnp

def get_signal_patch_positions(image, patch_size, stride, background_threshold):
    """
    Get signal patch positions using JAX.
    
    Note: This uses a two-pass approach - first count, then extract.
    """
    i_height, i_width = image.shape
    p_height, p_width = patch_size
    
    # Create grid of all possible positions
    rows = jnp.arange(0, i_height - p_height + 1, stride)
    cols = jnp.arange(0, i_width - p_width + 1, stride)
    
    # Vectorized patch extraction and checking
    def check_patch(r, c):
        patch = jax.lax.dynamic_slice(image, (r, c), (p_height, p_width))
        return jnp.max(patch) >= background_threshold
    
    # Use vmap to check all patches
    check_row = jax.vmap(lambda c: check_patch(0, c))
    check_all = jax.vmap(lambda r: jax.vmap(lambda c: check_patch(r, c))(cols))
    
    # Get mask of valid positions
    valid_mask = check_all(rows).flatten()
    
    # Create position grid
    r_grid, c_grid = jnp.meshgrid(rows, cols, indexing='ij')
    all_positions = jnp.stack([r_grid.flatten(), c_grid.flatten()], axis=1)
    
    # Filter by mask
    signal_positions = all_positions[valid_mask]
    
    # Check if any signal patches found
    if signal_positions.shape[0] == 0:
        raise ValueError("Couldn't find any signal patches!")
    
    return signal_positions
```

### aggregation.py Migration

```python
# Before (Numba with parallel)
from numba import jit, prange

@jit(nopython=True, parallel=True)
def aggregate_patches(estimate_denoised_image, weights, hyper_block, hyper_block_index):
    num_blocks, num_patches, ph, pw = hyper_block.shape
    for i in prange(num_blocks):
        for p in range(num_patches):
            patch = hyper_block[i, p]
            i_pos, j_pos = hyper_block_index[i, p]
            for ii in range(ph):
                for jj in range(pw):
                    estimate_denoised_image[i_pos + ii, j_pos + jj] += patch[ii, jj]
                    weights[i_pos + ii, j_pos + jj] += 1

# After (JAX with vmap)
import jax
import jax.numpy as jnp

@jax.jit
def aggregate_patches(estimate_denoised_image, weights, hyper_block, hyper_block_index):
    """
    Aggregate patches into the denoised image using JAX.
    
    Uses vmap for parallelization and functional updates.
    """
    num_blocks, num_patches, ph, pw = hyper_block.shape
    
    def add_single_patch(carry, patch_data):
        image, w = carry
        patch, pos = patch_data
        i_pos, j_pos = pos
        
        # Create indices for update
        i_indices = jnp.arange(i_pos, i_pos + ph)
        j_indices = jnp.arange(j_pos, j_pos + pw)
        ii, jj = jnp.meshgrid(i_indices, j_indices, indexing='ij')
        
        # Update image and weights
        image = image.at[ii, jj].add(patch)
        w = w.at[ii, jj].add(1)
        
        return (image, w), None
    
    # Process all blocks and patches
    for i in range(num_blocks):
        patches = hyper_block[i]
        positions = hyper_block_index[i]
        (estimate_denoised_image, weights), _ = jax.lax.scan(
            add_single_patch,
            (estimate_denoised_image, weights),
            (patches, positions)
        )
    
    return estimate_denoised_image, weights

# Alternative: Fully vectorized version (more complex but faster)
@jax.jit
def aggregate_patches_vectorized(estimate_denoised_image, weights, hyper_block, hyper_block_index):
    """
    Fully vectorized aggregation (more memory intensive but faster).
    """
    num_blocks, num_patches, ph, pw = hyper_block.shape
    
    # Flatten all patches and positions
    all_patches = hyper_block.reshape(-1, ph, pw)
    all_positions = hyper_block_index.reshape(-1, 2)
    
    # For each pixel in each patch, compute its position in the image
    def process_patch(patch, pos):
        i_pos, j_pos = pos
        return patch, i_pos, j_pos
    
    # Use at[].add() with index arrays
    # This is complex - may need custom implementation
    # based on specific requirements
    
    return estimate_denoised_image, weights
```

## Testing Strategies

### Numerical Validation

```python
import jax.numpy as jnp
import numpy as np

def test_numerical_equivalence():
    """Test that JAX and NumPy implementations give same results."""
    # Setup
    input_data = np.random.rand(100, 100)
    
    # NumPy version
    np_result = numpy_implementation(input_data)
    
    # JAX version
    jax_result = jax_implementation(jnp.array(input_data))
    jax_result_np = np.array(jax_result)
    
    # Compare
    np.testing.assert_allclose(np_result, jax_result_np, rtol=1e-6, atol=1e-6)
```

### Performance Benchmarking

```python
import time
import jax
import jax.numpy as jnp
import numpy as np

def benchmark_implementations():
    """Benchmark JAX vs NumPy implementations."""
    input_data = np.random.rand(1000, 1000)
    
    # NumPy benchmark
    start = time.time()
    for _ in range(100):
        _ = numpy_implementation(input_data)
    numpy_time = time.time() - start
    
    # JAX benchmark (including compilation)
    jax_input = jnp.array(input_data)
    start = time.time()
    _ = jax_implementation(jax_input)  # Compilation
    compilation_time = time.time() - start
    
    # JAX benchmark (without compilation)
    start = time.time()
    for _ in range(100):
        _ = jax_implementation(jax_input)
    jax_time = time.time() - start
    
    print(f"NumPy time: {numpy_time:.4f}s")
    print(f"JAX compilation time: {compilation_time:.4f}s")
    print(f"JAX execution time: {jax_time:.4f}s")
    print(f"Speedup: {numpy_time / jax_time:.2f}x")
```

## Performance Optimization

### Tips for Fast JAX Code

1. **Use JIT Compilation**
```python
@jax.jit
def fast_function(x):
    return jnp.sum(x ** 2)
```

2. **Minimize Python Loops**
```python
# Slow
result = 0
for i in range(len(arr)):
    result += arr[i] ** 2

# Fast
result = jnp.sum(arr ** 2)
```

3. **Use Static Arguments**
```python
@jax.jit
def my_function(x, static_value):
    # static_value causes recompilation if it changes
    return x * static_value

# Better
@functools.partial(jax.jit, static_argnums=(1,))
def my_function(x, static_value):
    return x * static_value
```

4. **Batch Operations**
```python
# Process items in batch
process_batch = jax.vmap(process_single)
results = process_batch(inputs)
```

## Common Pitfalls

### 1. Forgetting About Immutability

```python
# Wrong
arr[0] = 5

# Correct
arr = arr.at[0].set(5)
```

### 2. Using Python Control Flow in JIT

```python
# Wrong - Python if in JIT function
@jax.jit
def bad_function(x, threshold):
    if x > threshold:  # Python control flow
        return x
    return 0

# Correct - JAX control flow
@jax.jit
def good_function(x, threshold):
    return jax.lax.cond(x > threshold, lambda: x, lambda: 0)

# Or simpler
@jax.jit
def simple_function(x, threshold):
    return jnp.where(x > threshold, x, 0)
```

### 3. Not Splitting Random Keys

```python
# Wrong - reusing same key
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
y = jax.random.normal(key, (10,))  # Same as x!

# Correct - split keys
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, (10,))
key, subkey = jax.random.split(key)
y = jax.random.normal(subkey, (10,))  # Different from x
```

### 4. Dynamic Shapes in JIT

```python
# Wrong - shape depends on input value
@jax.jit
def bad_function(x, n):
    return jnp.zeros(n)  # Causes recompilation

# Correct - use static argument
@functools.partial(jax.jit, static_argnums=(1,))
def good_function(x, n):
    return jnp.zeros(n)
```

## Device Management

### Explicit Device Placement

```python
import jax

# Check available devices
devices = jax.devices()
print(f"Available devices: {devices}")

# Place array on specific device
cpu = jax.devices('cpu')[0]
gpu = jax.devices('gpu')[0] if jax.devices('gpu') else cpu

x_cpu = jax.device_put(x, cpu)
x_gpu = jax.device_put(x, gpu)

# Check array device
print(f"Array device: {x_gpu.device()}")
```

### Multi-GPU Support

```python
import jax

# Use pmap for multi-GPU parallelism
@jax.pmap
def parallel_function(x):
    return x ** 2

# Replicate input across devices
n_devices = jax.local_device_count()
x = jnp.arange(n_devices * 100).reshape(n_devices, 100)

# Compute on all devices in parallel
result = parallel_function(x)
```

## Conclusion

This technical guide provides the foundation for migrating BM3D-ORNL to JAX. Each module will require careful attention to the patterns described here, with thorough testing to ensure correctness and performance. The key is to embrace JAX's functional programming paradigm while maintaining the algorithmic integrity of the BM3D algorithm.
