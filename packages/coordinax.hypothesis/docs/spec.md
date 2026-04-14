# coordinax.hypothesis Specification

This document is the **normative** specification for the `coordinax.hypothesis` package, which provides Hypothesis strategies for generating valid `coordinax` objects (charts, roles, vectors, fiber points, etc.) for property-based testing.

This spec is intentionally **subordinate** to the core Coordinax spec:

- **Primary reference:** `coordinax/docs/spec.md` (the Coordinax Core Specification)
- `coordinax.hypothesis` must **not** redefine mathematics or semantics that already exist in Coordinax. Instead, it must generate test objects that satisfy the invariants and transformation laws described there.

---

## Goals

1. **Generate only valid objects** according to `coordinax/docs/spec.md`.
2. **Expose composable strategies** so tests can quantify over many charts/roles/metrics/embeddings.
3. **Support both broad and targeted testing**:
   - broad: quantify over all charts and roles to catch regressions,
   - targeted: generate specific families (e.g. Euclidean charts only; embedded charts only; physical tangent roles only).
4. **Be JAX-friendly** in the sense that generated values should be compatible with JAX tracing where applicable (e.g., array-like scalars, dtype stability). Strategy generation itself is not traced, but produced objects should be usable under `jax.jit` in tests.

## Writing Strategies with Multiple Dispatch

This section defines the required pattern for combining `plum.dispatch` with `hypothesis.strategies.composite` in `coordinax.hypothesis`.

### Required decorator order

When a strategy factory is both dispatched and composite, implementations MUST use this stacking order:

```text
@dispatch          # sees the public signature (draw already removed)
@st.composite      # strips draw; injects it at call-time
def my_strategy(draw, x: SomeType, ...):
    ...
```

Contract:

- `@st.composite` removes `draw` from the public call signature.
- Outer `@dispatch` therefore dispatches on user-facing typed arguments only.
- An annotation on `draw` (for example `draw: Any`) has no dispatch effect.

### Annotation rule

Files that define `@dispatch` + `@st.composite` strategy overloads MUST NOT use:

```python
from __future__ import annotations
```

Rationale:

- Postponed annotations are stored as strings in this mode.
- This breaks overload registration/equality behavior used by plum for this pattern.

### Varargs guidance

- Homogeneous varargs overloads (`*xs: int` vs `*xs: float`) are valid and dispatch correctly when all positional varargs satisfy one declared element type.
- Mixed-type varargs are not matched by homogeneous overloads. For heterogeneous varargs, dispatch per element inside one composite strategy.

---

## API

### Modules

- `coordinax.hypothesis.main`: the main entry point, with general-purpose strategies for common objects.
- `coordinax.hypothesis.angles`: strategies for generating various types of angular quantities.
- `coordinax.hypothesis.distances`: strategies for generating various types of distance quantities.

### Main module: `coordinax.hypothesis.main`

This module provides general-purpose strategies for generating valid `coordinax` objects, including:

| Module | Objects |
| --- | --- |
| `coordinax.hypothesis.angles` | `angles` |
| `coordinax.hypothesis.distances` | `distances` |
| `coordinax.hypothesis.charts` | `chart_classes`, `chart_init_kwargs`, `charts`, `charts_like`, `cdicts` |

### `coordinax.hypothesis.angles`

!!! info `angles`:

    Generate `unxt.Angle` values.

    Source:
    - `coordinax.hypothesis.angles.angles` is re-exported from `unxt_hypothesis.angles`.

    Signature:
    - `angles(*, wrap_to=None, **kwargs)`

    Parameters:
    - `wrap_to`: `None | tuple[min, max] | SearchStrategy[...]`. When provided, generated angles are wrapped to the specified interval. Bounds are quantities (`u.Q`) and can be strategy-valued.
    - `**kwargs`: forwarded to quantity generation. Common options include `unit`, `shape`, `dtype`, `elements`, `unique`.

    Contract:
    - Returns `u.Angle` instances.
    - If `unit` is omitted, defaults to angle dimension.
    - `quantity_cls` is fixed to `u.Angle` by the strategy.
    - Supports scalar and array-shaped outputs via `shape`.

    Failure behavior:
    - Invalid unit/parameter combinations follow `unxt_hypothesis.quantities` validation and raise from that implementation.

    Examples:
    - `@given(a=cxst.angles())`
    - `@given(a=cxst.angles(unit="rad"))`
    - `@given(a=cxst.angles(wrap_to=(u.Q(0, "deg"), u.Q(360, "deg"))))`

### `coordinax.hypothesis.distances`

!!! info `distances`:

    Generate `coordinax.distances.Distance` values.

    Source:
    - `coordinax.hypothesis.distances.distances` is implemented in `coordinax.hypothesis.distances._src.dist`.

    Signature:
    - `distances(*, check_negative=True, **kwargs)`

    Parameters:
    - `check_negative`: `bool | SearchStrategy[bool]`. `True` (default) enforces non-negative generated values.
    - `**kwargs`: forwarded to quantity generation. Common options include `unit`, `shape`, `dtype`, `elements`, `unique`.

    Contract:
    - Returns `coordinax.distances.Distance` instances.
    - If `unit` is omitted, defaults to length dimension.
    - `quantity_cls` is fixed to `coordinax.distances.Distance`.
    - Supports scalar and array-shaped outputs via `shape`.
    - Registers `st.from_type(coordinax.distances.Distance)` to this strategy.

    Non-negativity behavior:
    - If `check_negative=True` and `elements` is a mapping, `min_value` is
          raised to at least `0`.
    - If `check_negative=True` and `elements` is a strategy, values are mapped
          with `abs`.
    - If `check_negative=True` and `elements` is omitted, a default
          non-negative elements strategy is created.

    Failure behavior:
    - Invalid unit/parameter combinations follow
          `unxt_hypothesis.quantities` validation and raise from that
          implementation.

    Examples:
    - `@given(d=cxst.distances())`
    - `@given(d=cxst.distances(check_negative=False))`
    - `@given(d=cxst.distances(unit="kpc", shape=(2, 3)))`

### `coordinax.hypothesis.charts`

!!! info `chart_classes`:

    Draw chart classes (not instances) from subclasses of
    `coordinax.charts.AbstractChart`.

    Signature:
    - `chart_classes(filter=object, exclude_abstract=True, exclude=())`

    Parameters:
    - `filter`: `type | tuple[type, ...] | SearchStrategy[...]`.
      Tuple filters use AND semantics (must satisfy all).
    - `exclude_abstract`: `bool | SearchStrategy[bool]`.
      `True` (default) returns only concrete classes.
    - `exclude`: `tuple[type, ...]`.
      Covariant exclusion (excluded classes and their subclasses).

    Contract:
    - Returns `type[coordinax.charts.AbstractChart]`.
    - Never returns the base class `AbstractChart`.
    - Uses `coordinax.hypothesis.utils.get_all_subclasses` for discovery.
    - Supports dynamic (strategy-valued) `filter` and `exclude_abstract`.

    Failure behavior:
    - No matches -> `UserWarning` from subclass discovery.
    - Effective strategy then contains `sampled_from(())` and cannot draw.

    Examples:
    - `@given(chart_cls=cxst.chart_classes())`
    - `@given(chart_cls=cxst.chart_classes(filter=cxc.Abstract3D))`
    - `@given(chart_cls=cxst.chart_classes(exclude_abstract=False))`

!!! info `chart_init_kwargs`:

    Returns keyword arguments for initializing a chart instance: `chart_cls(**kwargs)`.

    Signature:
    - `chart_init_kwargs(chart_class, *, ndim=None)`

    Parameters:
    - `chart_class`: chart class or strategy producing one.
    - `ndim`: optional dimensionality constraint, including strategy-valued inputs.

    Contract:
    - Resolves strategy-valued inputs per draw.
    - For concrete chart classes, inspects required `__init__` parameters and draws annotation-driven kwargs.
    - For zero-required-parameter classes, returns `{}`.
    - `CartesianProductChart` has a specialized overload that returns `factors` and `factor_names` consistent with `ndim` when provided.

    Failure behavior:
    - Required init parameters without annotations raise `ValueError`.

!!! info `charts`:

    Generate chart instances (not classes).

    Signature:
    - `charts(chart_cls=None, /, filter=(), *, exclude=(Abstract0D, SphericalTwoSphere), ndim=None)`

    Parameters:
    - `chart_cls`: optional first positional argument. May be `None`, a chart class, or a strategy producing chart classes.
    - `filter`: class/tuple/strategy for class constraints.
    - `exclude`: tuple of excluded classes.
    - `ndim`: exact ndim, none, or strategy-valued ndim.

    Contract:
    - `chart_cls=None`: draws a chart class via `chart_classes(...)`, then redispatches on that class.
    - `chart_cls` as a strategy: draws the chart class first, then redispatches.
    - `chart_cls` as a chart class: draws kwargs via `chart_init_kwargs(...)`, constructs the instance, and enforces `ndim` with `assume(...)` when needed.
    - Abstract chart classes remain valid in the positional slot and act as chart-class filters.
    - Dimensional flag types such as `Abstract3D` are not chart classes and must still be passed via `filter=`.
    - If `ndim` is a concrete dimensional flag value, excludes other dimensional-flag families before class draw.
    - Specialized overloads cover `CartesianProductChart`, `SpaceTimeCT`, and `AbstractCartesianProductChart`.

    Specialized `charts(...)` overloads:

    - `charts(CartesianProductChart, /, *, factor_charts=None, factor_names=None, ndim=None, min_factors=1, max_factors=3)`
        - Generates general namespaced Cartesian product charts.
        - User-provided `factor_charts` and `factor_names` force the general `CartesianProductChart` path.
        - Otherwise generates factors internally and default names `f0`, `f1`, ...
        - Enforces `len(factors) == len(names)` with `assume(...)`.

    - `charts(SpaceTimeCT, /, *, ndim=None)`
        - Draws `spatial_chart` with `ndim - 1` (default total ndim is `4`).
        - Uses the default `c` most of the time; with low probability draws a random physically valid static speed quantity.

    - `charts(AbstractCartesianProductChart, /, *, factor_charts=None, factor_names=None, ndim=None, min_factors=1, max_factors=3)`
        - If factors/names are user-provided: generates the general `CartesianProductChart` path.
        - If factors/names are omitted: weighted selection between specialized flat-key product subclasses and general namespaced products.
        - Preserves requested total `ndim` when provided.

!!! info `charts_like`:

    Generate charts compatible with a template chart.

    Signature:
    - `charts_like(chart)`

    Contract:
    - Accepts a template chart or chart strategy.
    - Extracts dimensional flags from template MRO and draws a new chart with matching flags and `ndim`.
    - Uses `assume(...)` to keep only charts mutually transition-compatible with the template via chart realization/cartesian availability checks.

!!! info `cdicts`:

    Generate component dictionaries matching a chart schema.

    Signature:
    - `cdicts(chart, *, dtype=jnp.float32, shape=(), elements=None)`

    Parameters:
    - `chart`: chart instance or strategy producing one.
    - `dtype`, `shape`, `elements`: forwarded to quantity generation.

    Contract:
    - Returns a mapping whose keys are exactly `chart.components`.
    - Values are generated as quantities whose units follow `chart.coord_dimensions` component-by-component.
    - Supports scalar and array-valued payloads.
