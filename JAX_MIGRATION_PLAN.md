# JAX Migration Plan for BM3D-ORNL

## Executive Summary

This document outlines a comprehensive plan for migrating the BM3D-ORNL library from its current implementation using NumPy, Numba, and CuPy to using JAX as the core computing library. JAX offers unified CPU/GPU execution, automatic differentiation, JIT compilation, and vectorization capabilities that can simplify the codebase while maintaining or improving performance.

## Current Architecture Analysis

### Dependencies
- **NumPy**: Base array operations
- **Numba**: JIT compilation for CPU optimization (used in `utils.py` and `aggregation.py`)
- **CuPy**: GPU acceleration (used in `gpu_utils.py`)
- **SciPy**: Signal processing and interpolation

### Core Modules

#### 1. `gpu_utils.py` (CuPy-dependent)
- `hard_thresholding()`: FFT-based denoising with hard thresholding
- `wiener_hadamard()`: Wiener filtering using Hadamard transform
- `memory_cleanup()`: GPU memory management

#### 2. `utils.py` (Numba-dependent)
- `find_candidate_patch_ids()`: Spatial distance filtering (JIT compiled)
- `is_within_threshold()`: Euclidean distance comparison (JIT compiled)
- `get_signal_patch_positions()`: Patch extraction from images (JIT compiled)
- `pad_patch_ids()`: Array padding utilities
- `horizontal_binning()`: Multi-scale image processing
- `horizontal_debinning()`: Image interpolation (uses SciPy)
- `estimate_noise_std()`: Noise estimation

#### 3. `aggregation.py` (Numba-dependent)
- `aggregate_patches()`: Parallel patch aggregation (JIT compiled with `prange`)

#### 4. `block_matching.py` (NumPy-dependent)
- `PatchManager`: Manages patch extraction and grouping
- Depends on Numba functions from `utils.py`

#### 5. `denoiser.py` (Orchestration)
- `BM3D`: Main denoising class
- `bm3d_streak_removal()`: Multi-scale denoising pipeline
- Depends on all other modules

## Why JAX?

### Advantages

1. **Unified CPU/GPU Execution**: Single codebase for both CPU and GPU, eliminating the need for separate CuPy and NumPy paths
2. **Automatic JIT Compilation**: Built-in `jax.jit` decorator replaces Numba without requiring `nopython` mode constraints
3. **Automatic Vectorization**: `jax.vmap` for efficient batch processing
4. **Automatic Differentiation**: Future extensibility for optimization and ML applications
5. **XLA Compilation**: Cross-platform optimized code generation
6. **Functional Programming Paradigm**: Encourages pure functions and immutability, improving code reliability
7. **Better Memory Management**: Automatic memory optimization through XLA
8. **Growing Ecosystem**: Active community and extensive library support

### Potential Challenges

1. **Functional Paradigm**: JAX arrays are immutable; requires refactoring in-place operations
2. **Random Number Generation**: Different RNG approach (requires explicit keys)
3. **Dynamic Shapes**: JAX prefers static shapes for best performance
4. **Learning Curve**: Team needs to understand JAX idioms
5. **SciPy Dependencies**: Some SciPy functions need JAX equivalents
6. **Debugging**: JIT compilation can make debugging more complex

## Migration Strategy

### Phase 1: Foundation and Setup (Week 1-2)

#### Tasks
1. **Dependency Management**
   - Add JAX and jaxlib to `environment.yml` and `pyproject.toml`
   - Keep NumPy, CuPy, and Numba temporarily for gradual migration
   - Document JAX version requirements and compatibility

2. **Development Environment**
   - Set up JAX with GPU support for development team
   - Create JAX-specific testing infrastructure
   - Document installation procedures for various platforms

3. **Utility Module Creation**
   - Create `jax_utils.py` with JAX equivalents of utility functions
   - Implement helper functions for array operations
   - Add device management utilities (CPU/GPU selection)

#### Deliverables
- Updated `environment.yml` with JAX dependencies
- `src/bm3dornl/jax_utils.py` with basic utilities
- Installation and setup documentation
- JAX compatibility testing framework

### Phase 2: Core GPU Operations Migration (Week 3-4)

#### Module: `gpu_utils.py` → `jax_transforms.py`

**Migration Tasks:**

1. **FFT Operations (`hard_thresholding`)**
   ```python
   # Current (CuPy)
   hyper_block = cp.fft.rfft2(hyper_block, axes=(1, 2, 3))
   
   # JAX equivalent
   hyper_block = jax.numpy.fft.rfft2(hyper_block, axes=(1, 2, 3))
   ```
   - Replace CuPy FFT with `jax.numpy.fft`
   - Use `jax.jit` for compilation
   - Remove manual GPU memory management

2. **Hadamard Transform (`wiener_hadamard`)**
   - Implement Hadamard matrix generation in JAX
   - Replace `cp.einsum` with `jax.numpy.einsum`
   - Use `jax.jit` for performance
   - Handle shape transformations functionally

3. **Memory Management**
   - Remove explicit `memory_cleanup()` (JAX handles this automatically)
   - Rely on JAX's automatic memory management

**Testing Strategy:**
- Create parallel test suite comparing outputs between CuPy and JAX implementations
- Validate numerical accuracy (within floating-point tolerance)
- Benchmark performance on CPU and GPU
- Test with various input shapes and sizes

#### Deliverables
- `src/bm3dornl/jax_transforms.py` module
- Comprehensive unit tests for all transform functions
- Performance benchmarks (CuPy vs JAX)
- Migration guide for this module

### Phase 3: Numba-Optimized Functions Migration (Week 5-6)

#### Module: `utils.py` → JAX-compatible version

**Migration Tasks:**

1. **JIT-Compiled Functions**
   ```python
   # Current (Numba)
   @jit(nopython=True)
   def find_candidate_patch_ids(signal_patches, ref_index, cut_off_distance):
       ...
   
   # JAX equivalent
   @jax.jit
   def find_candidate_patch_ids(signal_patches, ref_index, cut_off_distance):
       ...
   ```

2. **Specific Function Migrations:**

   a. `find_candidate_patch_ids()`
      - Replace NumPy operations with JAX equivalents
      - Handle list comprehensions (use `jax.numpy` array operations)
      - Use `jax.lax.scan` or `jax.vmap` for loops

   b. `is_within_threshold()`
      - Direct JAX translation (minimal changes)
      - Use `jax.numpy.linalg.norm`

   c. `get_signal_patch_positions()`
      - Most challenging due to dynamic list growth
      - Pre-allocate array or use JAX's dynamic update strategies
      - Consider using `jax.lax.scan` for iteration

   d. `pad_patch_ids()`
      - Replace NumPy array operations with JAX equivalents
      - Handle different padding modes with `jax.lax.switch` or conditionals

   e. `horizontal_binning()` and `horizontal_debinning()`
      - Replace NumPy slicing with JAX slicing
      - For `horizontal_debinning()`, replace `scipy.interpolate.RectBivariateSpline`
      - Consider `jax.scipy.ndimage.map_coordinates` or custom interpolation

   f. `estimate_noise_std()`
      - Straightforward JAX conversion
      - Use JAX array operations

**Challenges and Solutions:**

| Challenge | Solution |
|-----------|----------|
| Dynamic array sizes in `get_signal_patch_positions()` | Pre-compute array size or use fixed-size arrays with masking |
| SciPy interpolation in `horizontal_debinning()` | Implement custom bilinear/bicubic interpolation in JAX or use JAX alternatives |
| Random number generation in `pad_patch_ids()` | Use JAX's PRNG with explicit key management |

#### Deliverables
- Migrated `utils.py` functions to JAX
- Unit tests comparing Numba and JAX implementations
- Performance benchmarks
- Documentation on differences and usage

### Phase 4: Aggregation Module Migration (Week 7)

#### Module: `aggregation.py` → JAX-compatible version

**Migration Tasks:**

1. **Parallel Aggregation**
   ```python
   # Current (Numba with prange)
   @jit(nopython=True, parallel=True)
   def aggregate_patches(...):
       for i in prange(num_blocks):
           ...
   
   # JAX equivalent using vmap
   @jax.jit
   def aggregate_patches(...):
       # Use vmap for vectorization
       jax.vmap(aggregate_single_block)(blocks)
   ```

2. **Implementation Strategy:**
   - Refactor nested loops into vectorized operations
   - Use `jax.vmap` for parallel processing
   - Ensure pure functional implementation (no in-place updates)
   - Use indexed updates with `.at[].add()` syntax

**Testing Strategy:**
- Verify identical results between Numba and JAX implementations
- Test with various block sizes and configurations
- Benchmark parallel performance

#### Deliverables
- JAX-based `aggregate_patches()` function
- Unit tests and performance benchmarks
- Documentation on parallelization strategy

### Phase 5: Block Matching Integration (Week 8)

#### Module: `block_matching.py` → JAX-compatible version

**Migration Tasks:**

1. **PatchManager Class Refactoring**
   - Maintain class structure but use JAX arrays internally
   - Replace NumPy operations with JAX equivalents
   - Ensure compatibility with JAX-based utility functions

2. **Specific Methods:**
   - `_generate_patch_positions()`: Use JAX-based `get_signal_patch_positions()`
   - `get_patch()`: Use JAX array slicing
   - `group_signal_patches()`: Use JAX operations for distance calculations
   - `get_hyper_block()`: Build blocks using JAX operations

3. **Design Considerations:**
   - Keep interface unchanged for backward compatibility
   - Internal operations use JAX
   - Consider making PatchManager functions instead of a class for better JAX integration

#### Deliverables
- Migrated `block_matching.py`
- Unit tests validating patch extraction and grouping
- Integration tests with JAX utilities

### Phase 6: Main Denoiser Integration (Week 9-10)

#### Module: `denoiser.py` → JAX-compatible version

**Migration Tasks:**

1. **BM3D Class Refactoring**
   - Update to use JAX-based modules
   - Replace NumPy arrays with JAX arrays
   - Ensure all operations are JAX-compatible

2. **Pipeline Integration:**
   - `thresholding()`: Use JAX transforms
   - `re_filtering()`: Use JAX transforms
   - `denoise()`: Orchestrate JAX-based pipeline

3. **Multi-scale Processing:**
   - Update `bm3d_streak_removal()` for JAX
   - Ensure compatibility with binning/debinning operations

#### Deliverables
- Fully migrated `denoiser.py`
- End-to-end integration tests
- Performance benchmarks (full pipeline)

### Phase 7: Testing and Validation (Week 11-12)

#### Comprehensive Testing

1. **Unit Tests**
   - Verify numerical equivalence between old and new implementations
   - Test edge cases and boundary conditions
   - Validate GPU and CPU execution

2. **Integration Tests**
   - End-to-end denoising pipeline tests
   - Multi-scale processing validation
   - Memory usage and performance profiling

3. **Regression Tests**
   - Compare outputs with reference implementation
   - Validate on standard test datasets
   - Check against published results

4. **Performance Benchmarks**
   - CPU vs GPU performance
   - JAX vs CuPy/Numba comparison
   - Memory consumption analysis
   - Scalability tests with different image sizes

#### Deliverables
- Comprehensive test suite
- Performance benchmark report
- Validation against reference implementation

### Phase 8: Documentation and Cleanup (Week 13-14)

#### Documentation Tasks

1. **Code Documentation**
   - Update docstrings for all migrated functions
   - Add JAX-specific usage examples
   - Document device selection and configuration

2. **User Documentation**
   - Update README with JAX requirements
   - Create migration guide for users
   - Add JAX installation instructions
   - Update example notebooks

3. **Developer Documentation**
   - Document JAX idioms and best practices
   - Create contributor guide for JAX development
   - Add troubleshooting section

#### Cleanup Tasks

1. **Dependency Removal**
   - Remove CuPy dependency (optional: keep for comparison)
   - Remove Numba dependency (optional: keep for comparison)
   - Update `environment.yml` and `pyproject.toml`

2. **Code Cleanup**
   - Remove deprecated code paths
   - Consolidate utility functions
   - Optimize import statements

#### Deliverables
- Complete documentation suite
- Updated dependencies
- Clean, production-ready codebase

## Implementation Guidelines

### JAX Best Practices

1. **Pure Functions**: All JAX functions should be pure (no side effects)
2. **Immutable Arrays**: Use `.at[].set()` for updates instead of in-place operations
3. **Static Shapes**: Prefer static shapes for best performance
4. **Explicit Randomness**: Use JAX PRNG with explicit key management
5. **Device Management**: Use `jax.devices()` for device selection
6. **JIT Compilation**: Use `@jax.jit` for performance-critical functions
7. **Vectorization**: Use `jax.vmap` instead of explicit loops

### Code Examples

#### Array Updates
```python
# Don't do this (in-place)
array[i] = value

# Do this (functional)
array = array.at[i].set(value)
```

#### Random Numbers
```python
# Don't do this
np.random.rand(10)

# Do this
key = jax.random.PRNGKey(0)
values = jax.random.uniform(key, shape=(10,))
```

#### Loops
```python
# Don't do this
result = []
for x in inputs:
    result.append(process(x))

# Do this
result = jax.vmap(process)(inputs)
```

## Performance Considerations

### Expected Benefits
- Unified codebase for CPU/GPU execution
- Potentially faster JIT compilation compared to Numba
- Better memory efficiency through XLA optimization
- Easier to maintain and extend

### Potential Concerns
- Initial JIT compilation overhead
- Learning curve for functional programming
- Some operations might need optimization for specific hardware

## Risk Mitigation

### Technical Risks

| Risk | Mitigation Strategy |
|------|-------------------|
| Performance regression | Comprehensive benchmarking at each phase; keep legacy code for comparison |
| Numerical accuracy issues | Rigorous testing with tolerance checks; validation against reference implementation |
| GPU compatibility | Test on multiple GPU architectures; maintain CPU fallback |
| Dynamic shape handling | Use padding and masking; document limitations |
| Team learning curve | Training sessions, documentation, code reviews |

### Project Risks

| Risk | Mitigation Strategy |
|------|-------------------|
| Timeline overrun | Phased approach allows early delivery; prioritize critical paths |
| Resource constraints | Parallel work where possible; clear task dependencies |
| Breaking changes | Maintain backward compatibility during transition; semantic versioning |
| Insufficient testing | Automated test suite; continuous integration |

## Success Criteria

1. **Functional Equivalence**: JAX implementation produces identical results (within numerical tolerance)
2. **Performance**: Comparable or better performance than CuPy/Numba implementation
3. **Code Quality**: Cleaner, more maintainable codebase
4. **Test Coverage**: >90% code coverage with JAX-based tests
5. **Documentation**: Complete user and developer documentation
6. **Backward Compatibility**: Smooth migration path for existing users

## Timeline Summary

- **Phase 1**: Foundation (2 weeks)
- **Phase 2**: GPU Operations (2 weeks)
- **Phase 3**: Numba Functions (2 weeks)
- **Phase 4**: Aggregation (1 week)
- **Phase 5**: Block Matching (1 week)
- **Phase 6**: Main Denoiser (2 weeks)
- **Phase 7**: Testing (2 weeks)
- **Phase 8**: Documentation (2 weeks)

**Total Estimated Time**: 14 weeks (3.5 months)

## Recommendations

### Immediate Actions

1. **Prototype**: Create a proof-of-concept for one critical function (e.g., `hard_thresholding`)
2. **Benchmark**: Establish baseline performance metrics with current implementation
3. **Training**: Team members should complete JAX tutorials and documentation
4. **Infrastructure**: Set up JAX development environment with GPU support

### Long-term Considerations

1. **Versioning**: Consider major version bump (e.g., 2.0.0) for JAX migration
2. **Maintenance**: Keep CuPy/Numba version in separate branch for comparison
3. **Extensions**: Leverage JAX for future features (e.g., gradient-based optimization)
4. **Community**: Engage with JAX community for best practices and support

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub Repository](https://github.com/google/jax)
- [JAX Ecosystem Projects](https://github.com/n2cholas/awesome-jax)
- [Common JAX Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [BM3D Algorithm Paper](https://doi.org/10.1109/TIP.2007.901238)

## Conclusion

Migrating BM3D-ORNL to JAX is a significant undertaking but offers substantial benefits in terms of code maintainability, performance portability, and future extensibility. The phased approach ensures minimal disruption while allowing for thorough testing and validation at each stage. With proper planning, team training, and execution, this migration can modernize the codebase and position it well for future enhancements.
