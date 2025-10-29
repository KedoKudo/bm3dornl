# JAX Migration FAQ

## General Questions

### Q1: Why are we migrating to JAX?

**A:** The migration to JAX offers several key benefits:

1. **Unified Codebase**: Single code for both CPU and GPU execution (no more separate NumPy/CuPy paths)
2. **Automatic Optimization**: XLA compiler provides better performance optimization
3. **Future-Proofing**: Built-in automatic differentiation enables future ML/optimization features
4. **Better Maintainability**: ~30% less code to maintain
5. **Automatic Memory Management**: No manual GPU memory cleanup needed

### Q2: Will my existing code break?

**A:** For most users, no. The external API will remain largely the same. However:

- You'll need to install JAX instead of (or in addition to) CuPy
- Some advanced usage patterns may require minor adjustments
- Performance characteristics might differ slightly

We will provide a detailed migration guide for users who have custom code built on BM3D-ORNL.

### Q3: What is the timeline for migration?

**A:** The planned timeline is approximately 14 weeks (3.5 months):
- Weeks 1-2: Foundation and setup
- Weeks 3-4: GPU operations migration
- Weeks 5-6: Numba functions migration
- Week 7: Aggregation module
- Week 8: Block matching
- Weeks 9-10: Main denoiser integration
- Weeks 11-12: Testing and validation
- Weeks 13-14: Documentation and cleanup

### Q4: Will there be a legacy version maintained?

**A:** Yes, we will maintain the current implementation in a separate branch during the transition period (6-12 months) to:
- Allow comparison and validation
- Provide fallback option if issues arise
- Give users time to migrate their workflows

## Technical Questions

### Q5: What are JAX's main advantages over NumPy/Numba/CuPy?

**A:** 

| Feature | Current (NumPy/Numba/CuPy) | JAX |
|---------|---------------------------|-----|
| CPU/GPU code | Separate paths | Unified |
| JIT compilation | Numba only | Built-in |
| Memory management | Manual (GPU) | Automatic |
| Differentiation | Not available | Automatic |
| Multi-device | Complex | Built-in |

### Q6: Will JAX be faster or slower?

**A:** Performance depends on the specific operation:

**Expected to be faster:**
- Operations involving multiple steps (XLA fusion optimization)
- Multi-device scenarios
- Complex pipelines

**Expected to be similar:**
- Simple FFT operations (both use cuFFT)
- Basic matrix operations (both use cuBLAS)

**Potentially slower:**
- First-time compilation (JIT overhead)
- Very small operations (compilation overhead not amortized)

Overall, we expect 5-15% performance improvement from XLA optimizations.

### Q7: Do I need a GPU to use JAX?

**A:** No. JAX works on both CPU and GPU. The same code runs on both:

```python
import jax.numpy as jnp

x = jnp.array([1, 2, 3])  # Runs on available device (CPU or GPU)
result = jnp.sum(x ** 2)   # Automatically optimized for device
```

### Q8: How does JAX handle random numbers differently?

**A:** JAX uses explicit random keys for reproducibility:

```python
# NumPy (implicit global state)
import numpy as np
np.random.seed(0)
x = np.random.randn(10)

# JAX (explicit key)
import jax
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10,))
```

This makes code more reproducible and easier to parallelize.

### Q9: What is the immutability requirement?

**A:** JAX arrays cannot be modified in place. Instead, you create new arrays:

```python
# NumPy (mutable)
arr[0] = 10

# JAX (immutable - functional update)
arr = arr.at[0].set(10)
```

This is required for JAX's transformations (jit, grad, vmap) to work correctly.

### Q10: Can I mix NumPy and JAX code?

**A:** Yes, but with caveats:

```python
import numpy as np
import jax.numpy as jnp

# Convert NumPy → JAX
np_array = np.array([1, 2, 3])
jax_array = jnp.array(np_array)

# Convert JAX → NumPy
jax_array = jnp.array([1, 2, 3])
np_array = np.array(jax_array)
```

However, mixing in computation-heavy code can cause performance issues due to data transfer overhead.

## Performance Questions

### Q11: Will compilation overhead be a problem?

**A:** For most use cases, no. JAX caches compiled functions, so you only pay compilation cost once:

```python
@jax.jit
def process(x):
    return x ** 2

# First call: compilation + execution (~100ms)
result = process(data)

# Subsequent calls: execution only (~1ms)
result = process(data)
```

For interactive use or functions called rarely, you can skip JIT.

### Q12: How much memory does JAX use?

**A:** JAX typically uses similar or less memory than CuPy because:
- XLA optimizes memory layout
- Automatic fusion reduces intermediate arrays
- No need for separate CPU/GPU copies

However, JIT compilation requires some additional memory for the compiled code.

### Q13: Can JAX run on multiple GPUs?

**A:** Yes, JAX has built-in support for multi-GPU:

```python
@jax.pmap
def parallel_process(x):
    return x ** 2

# Automatically distributes across all GPUs
n_devices = jax.local_device_count()
x = jnp.arange(n_devices * 100).reshape(n_devices, 100)
result = parallel_process(x)
```

## Migration Questions

### Q14: Do I need to change my code?

**A:** If you're using BM3D-ORNL as a library, minimal changes:

```python
# Before
from bm3dornl.denoiser import bm3d_streak_removal
result = bm3d_streak_removal(sinogram)

# After (same API)
from bm3dornl.denoiser import bm3d_streak_removal
result = bm3d_streak_removal(sinogram)
```

If you've extended or modified BM3D-ORNL, you may need to adapt your changes.

### Q15: Will old saved results still be valid?

**A:** Yes. The algorithm remains the same, so results should be numerically equivalent (within floating-point tolerance).

### Q16: What if I encounter a bug during migration?

**A:** We have several safeguards:
1. Comprehensive test suite to catch issues early
2. Legacy branch as fallback
3. Extensive validation against reference implementation
4. Community support and issue tracking

Please report any issues on our GitHub issue tracker.

## Installation Questions

### Q17: How do I install JAX?

**A:** 

```bash
# CPU-only
pip install jax jaxlib

# GPU (CUDA 12.x)
pip install jax[cuda12]

# Or using conda
conda install -c conda-forge jax
```

See [JAX installation guide](https://github.com/google/jax#installation) for more details.

### Q18: Does JAX work on my platform?

**A:** JAX supports:
- **Linux**: Full support (CPU, NVIDIA GPU, TPU)
- **macOS**: CPU only (GPU support experimental)
- **Windows**: CPU only via WSL2

For GPU support, you need:
- NVIDIA GPU with CUDA capability 5.0+
- CUDA 11.8+ or 12.x
- cuDNN 8.2+

### Q19: Can I use JAX with conda?

**A:** Yes:

```bash
conda install -c conda-forge jax jaxlib
```

For GPU support, you may need to install CUDA separately.

## Development Questions

### Q20: How can I contribute to the migration?

**A:** We welcome contributions! Here's how:

1. Check the [migration plan](JAX_MIGRATION_PLAN.md) for current status
2. Pick an unassigned task from the roadmap
3. Read the [technical guide](JAX_MIGRATION_TECHNICAL_GUIDE.md)
4. Submit a pull request with tests and documentation

### Q21: How do I test my JAX code?

**A:** Use pytest as usual:

```python
import jax.numpy as jnp
import numpy as np

def test_my_function():
    input_data = jnp.array([1, 2, 3])
    result = my_jax_function(input_data)
    expected = jnp.array([2, 4, 6])
    
    # Use numpy testing utilities
    np.testing.assert_allclose(result, expected, rtol=1e-6)
```

### Q22: How do I debug JAX code?

**A:** Several approaches:

1. **Disable JIT temporarily:**
```python
with jax.disable_jit():
    result = my_jitted_function(x)
```

2. **Use JAX debug print:**
```python
@jax.jit
def debug_func(x):
    jax.debug.print("x = {}", x)
    return x ** 2
```

3. **Enable NaN checking:**
```python
jax.config.update("jax_debug_nans", True)
```

### Q23: What IDE/editor should I use for JAX development?

**A:** Any Python-capable IDE works. Recommended:
- **VSCode**: Good JAX support with Python extension
- **PyCharm**: Excellent Python support
- **Jupyter**: Great for experimentation

Make sure to configure your IDE to recognize JAX's type hints.

## Compatibility Questions

### Q24: Is JAX compatible with other libraries I use?

**A:** JAX works well with:
- ✅ NumPy (seamless conversion)
- ✅ SciPy (has `jax.scipy` equivalent)
- ✅ Matplotlib (for visualization)
- ✅ PyTorch/TensorFlow (via conversion)

May have issues with:
- ❌ Libraries expecting mutable arrays
- ❌ Libraries with heavy Python control flow

### Q25: Can I use JAX with multiprocessing?

**A:** Yes, but with caveats:
- Use JAX's `pmap` for data parallelism
- Avoid Python's `multiprocessing` inside JIT functions
- See [JAX parallelism guide](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html)

### Q26: Does JAX work with Numba?

**A:** They can coexist but shouldn't be mixed:
- Don't call Numba functions from JAX (or vice versa)
- Use one or the other for a given function
- During migration, separate code paths are OK

## Advanced Questions

### Q27: Can I use automatic differentiation with BM3D?

**A:** After migration, yes! JAX provides `grad` and `value_and_grad`:

```python
@jax.jit
def loss_function(params, data):
    denoised = bm3d_with_params(data, params)
    return compute_loss(denoised, target)

# Compute gradient
grad_fn = jax.grad(loss_function)
gradients = grad_fn(params, data)
```

This enables optimization-based applications.

### Q28: What about custom CUDA kernels?

**A:** JAX supports custom operations:

```python
# Define custom XLA operation
from jax.lib import xla_client

@jax.custom_jvp
def my_custom_op(x):
    # Custom implementation
    pass
```

However, most operations don't need custom kernels due to XLA optimization.

### Q29: How does JAX handle large datasets?

**A:** Strategies for large data:
1. Use generators/iterators
2. Process in batches with `vmap`
3. Use `jax.device_put` strategically
4. Consider data parallelism with `pmap`

Example:
```python
@jax.jit
def process_batch(batch):
    return denoise(batch)

for batch in data_loader:
    result = process_batch(jnp.array(batch))
```

### Q30: Can JAX be used for production deployment?

**A:** Yes, JAX is production-ready:
- Used at Google, DeepMind, and others
- Stable API (1.0+ versions)
- Good performance and reliability
- Active maintenance and support

However, ensure proper testing and validation for your specific use case.

## Troubleshooting

### Q31: I'm getting "Tracer" errors. What's wrong?

**A:** This usually means you're using Python control flow in a JIT function:

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

### Q32: Why is my JAX code slow?

**A:** Common reasons:
1. **Recompilation**: Check if function is being recompiled repeatedly
2. **Data transfer**: Minimize host-device transfers
3. **Small arrays**: JIT overhead not amortized
4. **Python loops**: Use vectorization instead

Solution: Profile with `jax.profiler` and optimize bottlenecks.

### Q33: How do I report issues or get help?

**A:** Several channels:
1. **GitHub Issues**: For bugs and feature requests
2. **Discussions**: For questions and ideas
3. **JAX Discourse**: For JAX-specific questions
4. **Stack Overflow**: Tag with `jax`

## Resources

### Q34: Where can I learn more about JAX?

**A:** Recommended resources:
- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX 101 Tutorial](https://jax.readthedocs.io/en/latest/jax-101/index.html)
- [JAX GitHub](https://github.com/google/jax)
- [JAX Cookbook](https://github.com/google/jax/blob/main/docs/notebooks/Common_Gotchas_in_JAX.ipynb)

### Q35: Are there example projects using JAX?

**A:** Yes, many:
- **Flax**: Neural network library
- **Optax**: Gradient optimization library
- **JAXopt**: Optimization library
- **Diffrax**: Differential equations solver
- See [Awesome JAX](https://github.com/n2cholas/awesome-jax)

---

## Still Have Questions?

If your question isn't answered here:
1. Check the [JAX documentation](https://jax.readthedocs.io/)
2. Review our [migration plan](JAX_MIGRATION_PLAN.md) and [technical guide](JAX_MIGRATION_TECHNICAL_GUIDE.md)
3. Open an issue on our GitHub repository
4. Ask on our community discussion board

**Document Version:** 1.0  
**Last Updated:** 2025-10-29
