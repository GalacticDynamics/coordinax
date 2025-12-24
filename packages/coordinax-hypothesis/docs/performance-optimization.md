# Coordinax-Hypothesis Performance Optimization

## Summary

The `coordinax-hypothesis` strategies have been optimized using strategic
caching to significantly improve performance, particularly for the `charts()`
strategy.

## Key Optimizations

### 1. Cached `strategy_for_annotation` (utils.py)

Added `_cached_strategy_for_annotation()` wrapper with
`@ft.lru_cache(maxsize=256)`:

- Caches strategy generation for type annotations
- Prevents repeated strategy construction for common types (Quantity, Distance,
  Angle, etc.)
- Used in `build_init_kwargs_strategy()`

### 2. Cached `get_init_params` (utils.py)

Added `@ft.lru_cache(maxsize=128)` to `get_init_params()`:

- Caches parameter introspection for each representation class
- Avoids repeated `inspect.signature()` calls
- **Major performance improvement** (88% faster for simple representations)

### 3. Cached `build_init_kwargs_strategy` (representations.py)

Added caching wrapper with `@ft.lru_cache(maxsize=128)`:

- Caches entire strategy construction for each (class, dimensionality) pair
- Prevents re-building strategies for frequently used representation classes
- Works across all plum dispatches

### 4. Existing Cache

`get_all_subclasses()` was already cached with `@ft.cache` (no changes needed).

## Performance Results

### Before Optimization:

```
test_benchmark_build_init_kwargs_cart3d:      ~88 μs
test_benchmark_build_init_kwargs_spacetimect: ~242 μs
test_benchmark_get_all_subclasses:            ~211 ns (already cached)
test_benchmark_representations_*_build:       ~560 ns (fast, no change needed)
```

### After Optimization:

```
test_benchmark_build_init_kwargs_cart3d:      ~11 μs  (88% faster!) 🎉
test_benchmark_build_init_kwargs_spacetimect: ~236 μs (2% faster)
test_benchmark_get_all_subclasses:            ~211 ns (unchanged, already cached)
test_benchmark_representations_*_build:       ~560 ns (unchanged, already fast)
```

## Impact

The main improvement is for simple representations like `CartesianPos3D`:

- **Strategy construction**: 88% faster (88 μs → 11 μs)
- **Cache hit rate**: Very high since most tests use common representation
  classes repeatedly
- **User-facing improvement**: Tests using hypothesis strategies run noticeably
  faster

Complex recursive cases (like `SpaceTimeCT`) see smaller improvements (~2%)
because they still need to build nested strategies for their spatial components.

## Testing

New performance tests in `test_performance.py`:

- `test_representations_strategy_performance()`: Verifies strategy building is
  fast
- `test_chart_classes_performance()`: Tests class strategy building
- `test_drawing_representations_is_fast()`: End-to-end drawing performance

All tests pass, confirming optimizations don't break functionality.

## Caching Strategy

All caches use `functools.lru_cache` with reasonable sizes:

- `strategy_for_annotation`: 256 entries (many possible type annotations)
- `get_init_params`: 128 entries (one per representation class)
- `build_init_kwargs_strategy`: 128 entries (one per class+dimensionality pair)

These sizes are sufficient to cache all representation classes in `coordinax`
while remaining memory-efficient.

## Future Optimizations

Possible areas for further improvement:

1. Profile strategy _drawing_ (hypothesis internals) vs strategy _building_
2. Consider caching entire `charts()` strategy for common parameter combinations
3. Optimize nested/recursive strategy building (SpaceTimeCT, SpaceTimeEuclidean)
4. Consider pre-building strategies at module import time for ultra-fast access

## Files Modified

1. `/packages/coordinax-hypothesis/src/coordinax_hypothesis/_src/utils.py`:
   - Added `_cached_strategy_for_annotation()` wrapper
   - Added `@ft.lru_cache` to `get_init_params()`
   - Modified `build_init_kwargs_strategy()` to use cached annotation strategy

2. `/packages/coordinax-hypothesis/src/coordinax_hypothesis/_src/representations.py`:
   - Added caching wrapper for `build_init_kwargs_strategy()`
   - Imported `functools as ft`

3. `/packages/coordinax-hypothesis/tests/test_benchmark_strategies.py` (new):
   - Comprehensive benchmark suite using pytest-benchmark
   - Tests individual components and strategies

4. `/packages/coordinax-hypothesis/tests/test_performance.py` (new):
   - User-facing performance tests
   - Verifies strategies complete in reasonable time
