# Coordinax Core Specification

This document is the **normative** specification for the mathematical and software design of the `coordinax` coordinate and vector system. It defines a low-level, mathematically correct framework—**Charts**, **Metrics**, **Manifolds**, **Frames**, **Embeddings**, and **Maps**—and a high-level user API built on top of them -- **Point**, **Reference Frames**, **Coordinate**. The goals are:

1. **Correctness-first foundation**: rigorous definitions of points, tangents, their transformations, etc.
2. **Ergonomic high-level API**: common tasks should not expose low-level details.
3. **Extensibility via multiple dispatch**: functional API for existing functions are added by registration, not by editing core logic.
4. **JAX compatibility**: dispatch resolves on static Python objects; numerical kernels are pure and traceable.

## Table of Contents

```{contents}
:depth: 2
```

</br>

---

(math-spec)=

# The Math

In this section we fix the _mathematical objects_ that `coordinax` represents and the _transformation laws_ that determine how component data moves. This is a high-level overview; later sections specify each object and API in detail.

(math-spec-angles-and-distances)=

## Angles and Distances

<details>
<summary>Software Specification</summary>

- [Angles](#software-spec-angles)
- [Distances](#software-spec-distances)

</details>

A **quantity** is a pair $(x, u)$ where $x \in \mathbb{R}$ (or $\mathbb{R}^n$) is a numerical value and $u$ is a unit of measurement. Two quantities are physically equivalent when their values scale inversely with their units; unit conversion is a bijection on $\mathbb{R}$. _(Unit arithmetic is handled by `unxt`.)_

An **angular quantity** is special because it represents an element of the circle $S^1 = \mathbb{R}/2\pi\mathbb{Z}$, not of $\mathbb{R}$. $S^1$ is compact and has no boundary: there is no intrinsically "first" or "last" angle, and the space wraps. To store an angle as a real number one must choose a **branch cut** — a point excluded from the circle used as the interval endpoint. The two standard choices are $[0, 2\pi)$ and $(-\pi, \pi]$; both represent the same $S^1$. The cut is an artifact of representation: arithmetic that crosses it requires explicit wrapping, and the same physical angle has different numerical values under the two conventions.

A **distance quantity** has dimensions of length and, in the strict metric sense, is non-negative: it measures the magnitude of a displacement. A **radial coordinate**, however, can be signed — either because the coordinate system parameterizes a full line ($r \in \mathbb{R}$), or because a negative value is given physical meaning via the antipodal map $(r, \theta, \phi) \equiv (-r,\, \pi - \theta,\, \phi + \pi)$. The distinction between "distance" ($r \geq 0$) and "signed radial coordinate" ($r \in \mathbb{R}$) is a domain constraint on the same dimensional quantity. The point $r = 0$ is a coordinate singularity in spherical and cylindrical systems: the angular coordinates are degenerate there.

</br>

(math-spec-points)=

## Points

A point is an abstract geometric object. In this section we will explore how points can be concretized into coordinates -- e.g. "x", "y", "z" -- in coordinate representations --- e.g. Cartesian --- on manifolds -- e.g. $\mathbb{R}^3$. We will see how these coordinate representations of these points can be changed in the same manifold and across manifolds. We will also explore (time-dependent) actions on these coordinates, e.g. translations, and how to build these actions into changes of reference frames.

(math-spec-smooth-manifolds)=

### Smooth Manifolds

**Smooth** means that the manifold has a well-defined notion of differentiability, allowing us to perform calculus on it. **Manifold** indicates that the space is locally similar to $\mathbb{R}^n$ but can have a different global topology, such as being curved or having holes.

A **point** is simply an element $p \in M$.

Most of the important properties and structures associated with smooth manifolds --- such as coordinate systems, charts, atlases, and transition maps --- will be introduced and explained in subsequent sections.

(math-spec-charts)=

### Charts

[Software Spec](#software-spec-charts)

A smooth manifold is defined by its **atlas** $\mathcal{A}$ -- a collection of compatible charts used for representing coordinates. A **chart** is a local diffeomorphism[^diffeomorphism] assigning coordinates $q = (q^1,\dots,q^n)$ to points $p \in U$, where $U$ is the open subset of the manifold on which the coordinate system (chart) is valid. It is notated as the pair

$$
C=(U, \varphi),
$$

where

$$
\varphi : U \subset M \to \mathbb{R}^n,
$$

is the _chart map_ that performs the point to coordinate assignment.

!!! example

    As a simple example, take the 3-dimensional Spherical chart $C_s$ on a 3-dimensional Euclidean manifold $\mathbb{R}^3$. This chart covers the whole manifold, so we can write it simply as

    $$ C_s = (\mathbb{R}^3, \phi_s). $$

    Let $p$ be a point in $\mathbb{R}^3$.
    Applying the chart map $\phi_s$ to $p$ we get a coordinate $q$ with named components $(r,\theta,\phi)$ and physical dimensions of _(length,angle,angle)_.

[^diffeomorphism]: Formally, let M and N be smooth manifolds. A map $\tau : M \to N$ is a diffeomorphism if $\tau$ is bijective and smooth, and $\tau^{-1}$ is smooth. A diffeomorphism preserves dimension, smooth structure, and tangent bundle structures. It doesn't necessarily preserve the metric, distances, angles, or curvature.

### Atlases

An atlas $\mathcal{A}$ is a collection of compatible charts that cover the manifold

$$
\mathcal{A} = \{ (C_\alpha) \}.
$$

An atlas is defined through any collection of (compatible) charts; for example

$$
\mathcal{A} = \{ \text{Cartesian}, \text{Spherical} \}
$$

is a valid atlas for $\mathbb{R}^3$. The **maximal atlas** $\mathcal{A}_{\mathrm{max}}$ is the unique atlas containing every chart compatible with the original collection.

In [Smooth Manifolds](#math-spec-smooth-manifolds) we defined a smooth manifold as a topological space $M$ and a notion of differentiability. Now we can refine this definition, and specify that the maximal smooth atlas $\mathcal{A}_{\mathrm{max}}$ provides this differentiability notion. Thus a smooth manifold is a topological space equipped with a maximal smooth atlas --- formally $(M, \mathcal{A}_{\mathrm{max}})$, though often abbreviated back to just $M$.

(math-spec-transition-maps)=

### Transition Maps

[Software Spec](#software-spec-pt_map)

For charts in the same atlas of a manifold there are **transition maps** $\tau$ between any two charts for points on the manifold. These transition maps are composed of an inverse chart map that takes a coordinate $q$ back to the point $p$, followed by the forward chart map of the second chart that takes $p$ to the new coordinates $q'$. Transition maps are smooth and invertible, and they determine how point components transform between charts.

In mathematical notation, let chart $C_1 = (U, \varphi_1)$ and $C_2 = (V, \varphi_2)$, where $U, V$ have overlapping domains, then the transition map from chart 1 to 2 is

$$
\tau_{C_1\to C_2} \equiv \varphi_2 \circ \varphi_1^{-1} : \varphi_1(U \cap V) \to \varphi_2(U \cap V).
$$

!!! example

    Let Cartesian coordinates have chart $C_C = (\mathbb{R}^3, \varphi_C)$ and spherical coordinates have $C_S = (\mathbb{R}^3, \varphi_S)$. The transition map from Cartesian to spherical coordinates is

    $$
    \tau_{C\to S} \equiv \varphi_S(r, \theta, \phi) \circ \varphi_C^{-1}(x, y, z),
    $$

    which can be expressed numerically as:

    $$
    \tau_{C\to S} = \left( \sqrt(x^2+y^2+z^2), \arccos(z/r), \arctan(y/x) \right)
    $$

(math-spec-transition-maps-jacobian)=

#### Jacobian of the Transition Map

The **Jacobian** of a transition map is the matrix of first partial derivatives of the new coordinates with respect to the old coordinates. For

$$
\tau_{C_1\to C_2}(q) = q', \qquad q = (q^1,\dots,q^n), \quad q' = (\tilde{q}^1,\dots,\tilde{q}^n),
$$

the Jacobian at $q \in \varphi_1(U \cap V)$ is

$$
J^j{}_i(q)
= \frac{\partial \tau_{C_1\to C_2}^j}{\partial q^i}(q)
= \frac{\partial \tilde{q}^j}{\partial q^i}.
$$

Because $\tau_{C_1\to C_2}$ is a diffeomorphism on the overlap, $J(q)$ is invertible at every regular point of the overlap, and

$$
J(\tau_{C_1\to C_2}^{-1}(q')) = J(q)^{-1}
$$

for the inverse transition map.

</br>

---

# Packages

```{contents}
:local:
:depth: 1
```

## Main

This package collects all the most-commonly used parts of `coordinax` into a single convenient location.

It should be imported as `import coordinax.main as cx`

A non-exhaustive table of exported objects are:

| Package | Object |
| --- | --- |
| `coordinax.angles` | `AbstractAngle`, `Angle`, `wrap_to` |
| `coordinax.distances` | `AbstractDistance`, `Distance` |
| `coordinax.charts` | `CartesianProductChart`, </br> `cartesian_chart`, `guess_chart`, `cdict`, `pt_map`, `jacobian_pt_map`, `realize_cartesian`, </br> `cart0d`, </br> `cart1d`, `radial1d`, `time1d`, </br> `cart2d`, `polar2d`, </br> `cart3d`, `cyl3d`, `sph3d`, `lonlat_sph3d`, `loncoslat_sph3d`, `math_sph3d`, </br> `cartnd`, </br> `spacetimect` |
| `coordinax.representations` | `cconvert`, </br> `Representation`, `point`, </br> `PointGeometry`, `point_geom`, </br> `NoBasis`, `no_basis`, </br> `Location`, `loc`, </br> `guess_geometry_kind`, `guess_semantic_kind`, `guess_rep` |
| `coordinax.vectors` | `Point`, `ToUnitsOptions` |

</br>

## Internal

The `coordinax.internal` module exposes **semi-public** utilities intended for downstream packages and advanced integrations.

These names are importable and supported for inter-package use, but they are **not** part of the stable top-level `coordinax` API contract. Minor and patch releases may change names, signatures, or behavior in this module without warning. Code that depends on `coordinax.internal` should pin an exact version.

Semi-public API:

- `QuantityMatrix`: heterogeneous 1-D or 2-D quantity container with a per-element unit structure
- `UnitsMatrix`: immutable, hashable wrapper around a numpy object array of `AbstractUnit` elements, aligned with a `QuantityMatrix`; supports tuple-style indexing, iteration, and `to_tuple()`/`to_string()`. **Not** a subclass of `astropy.StructuredUnit`; bidirectional converters to/from `astropy.StructuredUnit` live in `coordinax.interop.astropy`.
- `cdict_units`: extract per-component units from a coordinate dictionary
- `pack_uniform_unit`: stack component data into an array using a shared unit
- `pack_nonuniform_unit`: stack component data into an array while preserving per-component units

</br>

(software-spec-angles)=

## Angles

The `coordinax.angles` module provides the angle-facing scalar API used by charts, vectors, and frame transformations.

!!! info `AbstractAngle`

    Abstract base type for angular quantities. It defines the shared angle interface used by dispatch and typing.

!!! info `Angle`

     Concrete angular scalar type (value + angular unit), built on `unxt`. Angles represent directions on $S^1$ and do not encode branch-cut convention in the type itself.

    - `wrap_to` method that calls `coordinax.angles.wrap_to` on the `Angle`.

!!! info `wrap_to`

    Functional API for explicit interval wrapping. It remaps an angle into a caller-specified interval (for example $[0, 2\pi)$ or $(-\pi, \pi]$).

!!! info `Parallax`

    Re-exported from `coordinax.distances` for convenience. `Parallax` is an angular quantity that is commonly used as a distance proxy in astronomy.

</br>

(software-spec-distances)=

## Distances

The `coordinax.distances` module provides the distance-facing scalar API used by coordinate charts and astronomy-oriented conversions.

!!! info `AbstractDistance`

    Abstract base type for distance-like quantities. It defines the shared interface used by dispatch and typing.

    Properties:

    - `distance`: convert to a [`coordinax.distances.Distance`](#software-spec-distance) with length units. This is the canonical distance representation for all distance-like quantities.

(software-spec-distance)=

!!! info `Distance`

    Concrete length quantity type (value + length unit), built on `unxt`, for physical distances.

!!! info `DistanceModulus`

    Magnitude-space distance representation used in astronomical workflows and conversions.

!!! info `Parallax`

    Angular distance proxy represented in angular units and linked to distance conversion flows.

</br>

(software-spec-charts)=

## Charts

The `coordinax.charts` module provides the chart-facing API for representing points on manifolds with explicit component names, ordering, and physical dimensions, including chart construction, chart selection, and transition maps between compatible coordinate representations.

!!! info `AbstractChart`

    ``AbstractChart`` is the base chart interface: concrete charts are specializations of this type.

    At its core, a chart defines a component schema -- names and order, e.g. _r, theta, phi_ -- and the physical dimensions of each coordinate component -- e.g. _length, angle, angle_.

    Public API:

    - ``components``: a tuple of the component names.
    - ``coord_dimensions``: a tuple of the dimensions (as strings) for points.
    - ``ndim``: number of coordinate components (chart dimensionality).
    - ``cartesian``: the corresponding global Cartesian  chart. This can raise an error if there isn't a global Cartesian chart. Calls `coordinax.charts.cartesian_chart`.
    - ``check_data()``: check that the data is compatible with the chart. Keyword arguments are ``keys`` (default ``True``) to validate key schema, and ``values`` (default ``False``) to validate value dimensions/ranges.
    - ``realize_cartesian``: realize a point in the canonical ambient Cartesian coordinates.
    - ``unrealize_cartesian``: invert the ambient Cartesian realization on the chart domain.

    Notes:

    - Registered to JAX as static using `jax.tree_util.register_static`.
    - Equality is based on matching classes, components, coord_dimensions, and all field values (if any).
    - Implements a `wadler_lindig` ``__pdoc__()`` method which underpins ``__repr__`` and ``__str__`

!!! info `AbstractFixedComponentsChart`

    ``AbstractFixedComponentsChart`` is a subclass of `AbstractChart` that makes it much easier to define a chart with fixed components and dimensions that cannot be changed on chart instantiation.

    ``AbstractFixedComponentsChart`` parses the type annotations to build the ``components`` and ``coord_dimensions`` attributes.

    For example, for a 1-D chart with component ``x``:

    <!-- skip: next -->

    ```python
    class ChartExample(
        AbstractFixedComponentsChart[tuple[Literal["x"]], Literal["Length"]]
    ):
        pass
    ```

!!! info `NoGlobalCartesianChartError`

    Raised when a chart has no global Cartesian representation.

    Some charts represent coordinates on curved manifolds (e.g., 2-sphere) that cannot be globally mapped to a flat Cartesian space without singularities or discontinuities.

### Functional API

!!! info `cartesian_chart`

    Return the canonical Cartesian chart for a chart object.

    `cartesian_chart` is a pure chart-selection operation: it returns a chart object only, and does not transform coordinate data. This function underpins `AbstractChart.cartesian`.

    - For Euclidean standard charts, return the Cartesian chart of matching coordinate dimensionality.
    - Mapping: `Abstract0D -> cart0d`, `Abstract1D -> cart1d`, `Abstract2D -> cart2d`, `Abstract3D -> cart3d`, `AbstractND -> cartnd`.
    - Special-case `Time1D`: `Time1D -> time1d` (time is already canonical in this 1D chart family).
    - For `CartesianProductChart`, apply `cartesian_chart` factorwise and preserve `factor_names`.
    - For `SpaceTimeCT`, apply `cartesian_chart` to the spatial factor and keep the `ct` time factor convention.
    - The operation is idempotent: `cartesian_chart(cartesian_chart(C)) == cartesian_chart(C)`.

    Failure semantics:

    - For intrinsic 2-sphere charts (`AbstractSphericalTwoSphere`), there is no global Cartesian 2D chart. The function raises `NoGlobalCartesianChartError`.

!!! info `guess_chart`

    Infer a chart from lightweight structural information.

    `guess_chart` provides heuristic chart inference for common inputs where full type information is unavailable.

    Dispatches:

    - Input `frozenset[str]` (component names): infer a chart whose `components` set matches exactly. The results are cached; repeated calls with the same key set return the same chart object instance.
    - Input `CDict`/mapping: infer from `frozenset(obj.keys())`.
    - Input array-like or quantity-like with trailing axis length 1, 2, or 3: infer Cartesian chart by that trailing size.
    - Mapping for shaped inputs: trailing `...x1 -> cart1d`, `...x2 -> cart2d`, `...x3 -> cart3d`.

    Failure semantics:

    - If no chart matches a provided key set, raise `ValueError`.
    - Inputs outside registered dispatches (for example trailing axis sizes other than 1/2/3) are not inferred by this API.

    Notes:

    - There is a selection ambiguity. Key-based inference uses component names only. If multiple chart types share the same component names, the result is not uniquely identifiable from keys alone and one matching chart is returned.
    - Inferred chart choice from keys is stable within a process due to caching, but callers should not treat key-only inference as a unique chart identifier when component-name collisions exist.

!!! info `cdict`

    Extract a component dictionary from an object.

    `cdict` decomposes vectors and coordinate data into dictionaries where keys are component names (from a chart) and values are the corresponding scalars or arrays. This is a normalization operation: multiple input types map to a common `CDict` representation.

    Dispatches:

    - Input `CDict` (dict with string keys): returned unchanged (identity).
    - Input `unxt.AbstractQuantity` with last axis size 1, 2, or 3: call `guess_chart` on the quantity to infer chart dimensionality, then apply the chart-based dispatch.
    - Input array-like with chart context: split last axis into named components using `chart.components`. Last axis length MUST match the chart's component count.
    - Input `unxt.AbstractQuantity` with chart context: split last axis into named quantities using `chart.components`. Requires last axis size to match chart.
    - Input `QuantityMatrix` with chart context: extract heterogeneous per-component quantities, one for each chart component.

    Failure semantics:

    - If last axis size does not match the number of chart components, raise `ValueError`.
    - If input type has no registered dispatch, `plum.NotFoundLookupError` is raised.

    Notes:

    - When no chart is explicitly provided, `guess_chart` is used to infer Cartesian chart dimensionality from array/quantity shape. This works for arrays/quantities with trailing axis size 1, 2, or 3 only.
    - Units are preserved: a quantity input returns a `CDict` of quantities.

(software-spec-pt_map)=

!!! info `pt_map`

    Transform point coordinates from one chart to another.

    This function implements the most general point-coordinate map between two compatible chart representations of the same geometric point. It is a point-wise map that preserves the physical location while changing the coordinate description.

    `pt_map` handles both same-manifold chart transitions (the ordinary chart transition map between two charts in the same atlas) and cross-manifold **realization-style** maps (between charts attached to different manifolds when one is a realization of the other, such as an intrinsic chart on an embedded manifold and a chart on its ambient manifold).

    Dispatches:

    - Input ``(p, from_chart, to_chart, usys=None)``: Transform coordinates in `p` from ``from_chart`` to ``to_chart``, representing the same underlying geometric point under a different coordinate representation.
    - Input (from_chart, to_chart, **fixed_kwargs): Partial application returning a callable that accepts `(p, *args, **kw)` and forwards to `pt_map(p, from_chart, to_chart, **fixed_kwargs, **kw)`.
    - Identity transformation: When `to_chart` and `from_chart` are the same chart type (e.g., both `Cart3D`), return `p` unchanged. The output is identical to the input dict.
    - Specific dispatches: Direct transformations exist for many chart pairs on the same manifold (e.g., `Cart3D` ↔ `Cylindrical3D`, `Cart3D` ↔ `Spherical3D`, `Polar2D` ↔ `Cart2D`, `Cart1D` ↔ `Radial1D`).
    - Fallback via Cartesian: If no specific dispatch exists for (from_chart, to_chart), use a two-step intermediate: ``pt_map(p, from_chart, to_chart.cartesian, usys=usys)`` followed by ``pt_map(p_cart, from_chart.cartesian, to_chart, usys=usys)``.
    - Product charts (factorwise): For `AbstractCartesianProductChart` ↔ `AbstractCartesianProductChart`, apply `pt_map` independently to each factor via `split_components` and `merge_components`.
    - Cross-manifold dispatches: Additional dispatches can define realization-style mappings between charts on different manifolds (e.g., an intrinsic chart on an embedded manifold to a chart on the ambient manifold).
    - The `usys` parameter (unit system) is used to interpret non-quantity inputs (e.g., bare floats) in appropriate units. It is optional for unit-ful quantities and required otherwise.

    Failure semantics:

    - If `to_chart` and `from_chart` do not have a dispatch path.
    - If `to_chart` or `from_chart` are product charts but the other is not.
    - If product charts have mismatched factor counts, raise `TypeError`.
    - For non-Quantity inputs without `usys` provided when required for interpretation.

    Notes:

    - This is a position-only transformation.
    - The partial-application dispatch (returning a callable) is useful for currying: `transform_func = pt_map(cart3d, sph3d); result = transform_func(p)`.
    - Product charts (like SpaceTimeCT with a spatial factor) transform by independently transforming each factor's coordinates.
    - Same-atlas chart transitions and cross-manifold realization maps are unified under one function; dispatch resolution selects the appropriate implementation based on the chart types.

!!! info `realize_cartesian`

    Realize point coordinates in a chart's canonical ambient Cartesian chart.

    This function evaluates the chart's ambient realization map by converting
    coordinates from `chart` into `chart.cartesian`.

    This is implemented as a thin wrapper over `pt_map`.

(software-spec-tangent-map)=

!!! info `jacobian_pt_map`

    Compute the Jacobian of the chart transition map at a base point.

    **Dispatches:**

    - `(at: None, /, *fixed_args, usys: AbstractUnitSystem, **fixed_kw)` -> partial
      application. Returns a callable that accepts `(at, *args, **kw)` and forwards to
      `jacobian_pt_map(at, *fixed_args, *args, **fixed_kw, **kw)`. Requires `usys`.

    - `(from_chart, to_chart, /, *, usys: AbstractUnitSystem)` -> curried partial
      application. Returns `lambda at: jacobian_pt_map(at, from_chart, to_chart,
      usys=usys)`. Requires `usys`. Useful for repeated Jacobian evaluation at multiple
      base points with a fixed unit system.

    - `(at: Array, from_chart, to_chart, /, *, usys: AbstractUnitSystem)` -> plain
      `Array`. Treats `at` as a flat numeric array (no units), computes
      `jax.jacfwd(pt_map(None, from_chart, to_chart, usys=usys))(at)`, and returns the
      raw array Jacobian. Requires `usys`.

    - `(at: CDict, from_chart, to_chart, /, *, usys: OptUSys = None)` ->
      `Array | QuantityMatrix`. The general dict dispatch. Branches on whether `at`
      values are plain arrays or quantities:

      - **Array-valued** (`is_array=True`): stacks `at` into a plain array via
        `jnp.stack`, then forwards to the `(at: Array, ...)` dispatch. `usys` is
        forwarded and must be an `AbstractUnitSystem` unless a more-specific analytical
        dispatch handles it.

      - **Quantity-valued** (`is_array=False`): packs `at` into a 1-D `QuantityMatrix`
        via `pack_to_qmatrix(at, keys=from_chart.components)`, casts to `float`, then
        computes `J_qq = jax.jacfwd(pt_map_fn)(at_in)`. The jacfwd result is a
        `QuantityMatrix` whose `.value` is itself a `QuantityMatrix` encoding the input
        units, and whose `.unit` encodes the output units. `_repack_q_from_jac` extracts
        both to build the correct 2-D `UnitsMatrix` and returns
        `QuantityMatrix(J_arr, unit=unit_matrix)` of shape `(n_out, n_in)`.

    **`usys` parameter:** required for the `None`-partial, curried, and plain-`Array`
    dispatches. Optional (`None`) for the `CDict` generic dispatch's quantity-valued
    branch, and for all analytical dispatches.

    **Analytical dispatches** (higher precedence than the generic `CDict` fallback;
    `usys` is optional):

    - `Cart2D -> Polar2D`: `Array`, `AbstractQuantity`, and `CDict` overloads.
      The `AbstractQuantity` overload computes closed-form partial derivatives and
      explicitly sets the ∂θ row units to `rad / input_length_unit` (astropy treats rad
      as dimensionless, so the unit must be forced manually).
    - Further analytical pairs (`Polar2D -> Cart2D`, `Cart3D ↔ Sph3D`,
      `Cart3D ↔ Cyl3D`) follow the same pattern via the generic `CDict` dispatch.

    **Failure semantics:**

    - Raises `plum.NotFoundLookupError` when calling the `(at: CDict, ...)` dispatch
      with `is_array=True` values and `usys=None` for a chart pair that has no analytical
      `Array` dispatch (e.g., `cart3d -> sph3d` without a unit system).
    - Raises `ValueError` if `at` keys do not match `from_chart.components` (via
      `check_data`).

### Dimensional Flags

!!! info `AbstractDimensionalFlag`

    ``AbstractDimensionalFlag`` is a mixin used to mark chart dimensionality (for example 0-D, 1-D, 2-D, 3-D, 6-D, and N-D categories).

    - These classes cannot be instantiated and MUST be combined with concrete subclasses of `AbstractChart`.
    - Registered to JAX as static using `jax.tree_util.register_static`.

!!! info `DIMENSIONAL_FLAGS`

    ``DIMENSIONAL_FLAGS`` is the registry of dimensional flags, populated as subclasses of `AbstractDimensionalFlag` are created.

    Subclasses can pass a class-level kwarg registering the dimensionality. For example:

    <!-- skip: next -->

    ```python
    class Abstract0D(AbstractDimensionalFlag, n=0):
        pass
    ```

!!! info `Abstract0D`

    Marker base class for 0-dimensional charts.

    - Semantics: represents coordinate systems with zero components (no coordinate keys).
    - Dimensional flag: `n = 0` only; variable `n` is not supported.
    - Subclass constraint: subclasses must also be chart subclasses; otherwise `TypeError`.
    - If a subclass tries to set `n` explicitly, raise `NotImplementedError`.

!!! info `Abstract1D`

    Marker base class for 1-dimensional charts.

    - Semantics: represents coordinate systems with one component (one coordinate keys).
    - Dimensional flag: `n = 1` only; variable `n` is not supported.
    - Subclass constraint: subclasses must also be chart subclasses; otherwise `TypeError`.
    - If a subclass tries to set `n` explicitly, raise `NotImplementedError`.

!!! info `Abstract2D`

    Abstract marker for 2-dimensional charts.

    - `Abstract2D` extends `AbstractDimensionalFlag` with `n=2`.
    - Concrete 2-D charts should subclass `Abstract2D`.
    - Does not imply flatness: e.g. the two-sphere uses two angular coordinates on a curved manifold.

!!! info `Abstract3D`

    Abstract marker for 3-dimensional charts.

    - `Abstract3D` extends `AbstractDimensionalFlag` with `n=3`.
    - Concrete 3-D charts should subclass `Abstract3D`.

!!! info `Abstract6D`

    Abstract marker for 6-dimensional charts.

    - `Abstract6D` extends `AbstractDimensionalFlag` with `n=6`.
    - Concrete 6-D charts should subclass `Abstract6D`.

!!! info `AbstractND`

    Abstract marker for N-dimensional charts.

    - `AbstractND` extends `AbstractDimensionalFlag` with `n=6`.
    - Concrete N-D charts should subclass `AbstractND`.

### Standard Euclidean Charts

#### 0-D Charts

!!! info `Cart0D` and `cart0d`

    Canonical concrete 0-D Cartesian chart and its instance.

    - `Cart0D` is the final concrete chart type for 0-dimensional coordinates.
    - It has no components and no component dimensions (`components=()`, `dimensions=()`).
    - `cart0d` is its pre-defined instance.
    - As the canonical 0D Cartesian chart, it is the target of `cartesian_chart(Abstract0D)`.

#### 1-D Charts

!!! info `Cart1D` and `cart1d`

    Canonical concrete 1-D Cartesian chart and its instance.

    - `Cart1D` is the final concrete chart type for 1-dimensional Cartesian coordinates.
    - Components: `("x",)` with dimension `("length",)`.
    - `cart1d` is its pre-defined instance.
    - As the canonical 1D Cartesian chart, it is the target of `cartesian_chart(Abstract1D)`.

!!! info `Radial1D` and `radial1d`

    Concrete 1-D radial chart and its instance.

    - `Radial1D` is the final concrete chart type for 1-dimensional radial coordinates.
    - Components: `("r",)` with dimension `("length",)`.
    - `radial1d` is its pre-defined singleton instance.
    - Semantically equivalent to `Cart1D` but uses `r` instead of `x`; transition between the two is a pure component rename (`r ↔ x`).
    - Cartesian projection is canonical: `cartesian_chart(Radial1D) -> cart1d`.

!!! info `Time1D` and `time1d`

    Concrete 1-D time chart and its pre-defined instance.

    - `Time1D` is the final concrete chart type for 1-dimensional time coordinates.
    - Components: `("t",)` with dimension `("time",)`.
    - `time1d` is its pre-defined `Time1D()` instance.
    - `Time1D` is its own Cartesian chart: `cartesian_chart(Time1D) -> time1d`.

#### 2-D Charts

!!! info `Cart2D` and `cart2d`

    Canonical concrete 2-D Cartesian chart and its pre-defined instance.

    - `Cart2D` is the final concrete chart type for 2-dimensional Cartesian coordinates.
    - Components: `("x", "y")` with dimensions `("length", "length")`.
    - `cart2d` is its pre-defined `Cart2D()` instance.
    - As the canonical 2-D Cartesian chart, it is the target of `cartesian_chart(Abstract2D)`.

!!! info `Polar2D` and `polar2d`

    Concrete 2-D polar chart and its pre-defined instance.

    - `Polar2D` is the final concrete chart type for 2-dimensional polar coordinates.
    - Components: `("r", "theta")` with dimensions `("length", "angle")`.
    - `polar2d` is its pre-defined `Polar2D()` instance.
    - Direct transitions registered: `Polar2D ↔ Cart2D` (via `r`, `theta` ↔ `x`, `y` trig formulas).
    - Cartesian projection is canonical: `cartesian_chart(Polar2D) -> cart2d`.

#### 3-D Charts

!!! info `Cart3D` and `cart3d`

    Canonical concrete 3-D Cartesian chart and its pre-defined instance.

    - `Cart3D` is the final concrete chart type for 3-dimensional Cartesian coordinates.
    - Components: `("x", "y", "z")` with dimensions `("length", "length", "length")`.
    - `cart3d` is its pre-defined `Cart3D()` instance.
    - As the canonical 3-D Cartesian chart, it is the target of `cartesian_chart(Abstract3D)`.

!!! info `Cylindrical3D` and `cyl3d`

    Concrete 3-D cylindrical chart and its pre-defined instance.

    - `Cylindrical3D` is the final concrete chart type for cylindrical coordinates.
    - Components: `("rho", "phi", "z")` with dimensions `("length", "angle", "length")`.
    - `cyl3d` is its pre-defined `Cylindrical3D()` instance.
    - Direct transitions registered: `Cylindrical3D ↔ Cart3D`.
    - Cartesian projection is canonical: `cartesian_chart(Cylindrical3D) -> cart3d`.

!!! info `AbstractSpherical3D`

    Abstract chart for 3-D spherical-family charts.

    - `AbstractSpherical3D` is a non-final intermediate base for all 3-D spherical charts.
    - Concrete spherical charts (`Spherical3D`, `LonLatSpherical3D`, `LonCosLatSpherical3D`, `MathSpherical3D`, `ProlateSpheroidal3D`) all subclass this.
    - Shares the `Abstract3D` Cartesian projection: `cartesian_chart(AbstractSpherical3D) -> cart3d`.

!!! info `Spherical3D` and `sph3d`

    Physics-convention 3-D spherical chart and its pre-defined instance.

    - `Spherical3D` is the final concrete chart type for physics-convention spherical coordinates $(r, \theta, \phi)$.
    - Components: `("r", "theta", "phi")` with dimensions `("length", "angle", "angle")`.
    - `theta` is the polar (colatitude) angle, $\theta \in [0, \pi]$; `phi` is the azimuth.
    - Embedding: $x = r\sin\theta\cos\phi$, $y = r\sin\theta\sin\phi$, $z = r\cos\theta$.
    - `check_data` enforces `theta` in valid polar range.
    - `sph3d` is its pre-defined `Spherical3D()` instance.

!!! info `LonLatSpherical3D` and `lonlat_sph3d`

    Longitude-latitude spherical chart and its pre-defined instance.

    - `LonLatSpherical3D` is the final concrete chart type for lon/lat/distance coordinates.
    - Components: `("lon", "lat", "distance")` with dimensions `("angle", "angle", "length")`.
    - `lat` is the latitude in $[-\pi/2, \pi/2]$; `lon` is the azimuth.
    - Relation to `Spherical3D`: $\mathrm{lat} = \pi/2 - \theta$, $\mathrm{lon} = \phi$.
    - `check_data` enforces `lat` in $[-90°, 90°]$.
    - `lonlat_sph3d` is its pre-defined `LonLatSpherical3D()` instance.

!!! info `LonCosLatSpherical3D` and `loncoslat_sph3d`

    Longitude-cos(latitude) spherical chart and its pre-defined instance.

    - `LonCosLatSpherical3D` is the final concrete chart type for $(\mathrm{lon}\cos\mathrm{lat}, \mathrm{lat}, r)$ coordinates.
    - Components: `("lon_coslat", "lat", "distance")` with dimensions `("angle", "angle", "length")`.
    - The `lon_coslat` component improves numerical behaviour near the poles where $\cos(\mathrm{lat}) \to 0$.
    - `check_data` enforces `lat` in $[-90°, 90°]$.
    - `loncoslat_sph3d` is its pre-defined `LonCosLatSpherical3D()` instance.

!!! info `MathSpherical3D` and `math_sph3d`

    Math-convention 3-D spherical chart and its pre-defined instance.

    - `MathSpherical3D` is the final concrete chart type for math-convention spherical coordinates $(r, \theta, \phi)$.
    - Components: `("r", "theta", "phi")` with dimensions `("length", "angle", "angle")`.
    - In this convention `phi` is the polar angle (from $z$-axis, $\phi \in [0°, 180°]$) and `theta` is the azimuth — the reverse of `Spherical3D`.
    - `check_data` enforces `phi` in $[0°, 180°]$.
    - `math_sph3d` is its pre-defined `MathSpherical3D()` instance.

!!! info `ProlateSpheroidal3D`

    Prolate spheroidal chart with focal parameter.

    - `ProlateSpheroidal3D` is the final concrete chart type for prolate spheroidal coordinates $(\mu, \nu, \phi)$.
    - Components: `("mu", "nu", "phi")` with dimensions `("area", "area", "angle")`.
    - Carries a required field `Delta` (focal half-length, a static `Quantity["length"]` with `Delta > 0`).
    - Validity constraints: $\mu \ge \Delta^2$, $|\nu| \le \Delta^2$.
    - Unlike many other charts, `ProlateSpheroidal3D` instances are not interchangeable: two instances with different `Delta` are distinct charts.
    - No pre-defined instance (requires `Delta` parameter).
    - Cartesian projection is canonical: `cartesian_chart(ProlateSpheroidal3D) -> cart3d`.

#### N-D Charts

!!! info `CartND` and `cartnd`

    Canonical concrete N-D Cartesian chart and its pre-defined instance.

    - `CartND` is the final concrete chart type for Cartesian coordinates in arbitrary dimension.
    - Components: `("q",)` with dimension `("length",)`, where `q` stores the N Cartesian components as a single length-valued array.
    - `cartnd` is its pre-defined `CartND()` instance.
    - `CartND` is already Cartesian: `cartesian_chart(CartND) -> cartnd`.

### Cartesian Product Charts

!!! info `AbstractCartesianProductChart`

    Abstract base class for Cartesian product charts.

    A Cartesian product chart represents a finite ordered tuple of factor charts: $$ M = \prod_i M_i $$ with factorwise coordinate transformation laws.

    Public API:

    - `factors: tuple[AbstractChart, ...]`: ordered factor charts.
    - `factor_names: tuple[str, ...]`: ordered factor names aligned with `factors` (same length; unique in concrete implementations).
    - `ndim`: total dimension, computed as `sum(f.ndim for f in factors)`.
    - `components` (default behavior): dot-delimited namespaced keys `("name0.c0", ..., "nameN.cK")`.
      - Flat-key product charts are an explicit specialization (`AbstractFlatCartesianProductChart`), used by `SpaceTimeCT`.
    - `coord_dimensions`: concatenation of factor coordinate dimensions in factor order.
    - `split_components(p)`: partitions a `CDict` into one per factor, stripping factor prefixes.
    - `merge_components(parts)`: re-attaches prefixes and merges factor dictionaries.

    Transform semantics:

    - Registered `pt_map` between two product charts is factorwise: split input by factors, transform each factor pair, then merge.
    - If factor counts differ, `pt_map` raises `TypeError`.
    - No automatic product↔non-product transform is defined; those paths raise `NotImplementedError` unless explicit dispatches are added.

!!! info `CartesianProductChart`

    Concrete namespaced Cartesian product chart.

    `CartesianProductChart` is the final implementation of
    `AbstractCartesianProductChart` for general products using dot-delimited
    keys.

    Construction:

    - Parameters:
      - `factors`: tuple of factor charts.
      - `factor_names`: tuple of names for each factor.
    - Post-init validation:
      - `len(factors) == len(factor_names)`, otherwise `ValueError`.
      - `factor_names` are unique, otherwise `ValueError`.

    Component model:

    - Keys are namespaced by factor name:
      e.g. factors `(cart3d, cart3d)` and names `("q", "p")` produce `("q.x", "q.y", "q.z", "p.x", "p.y", "p.z")`.
    - This avoids key collisions for repeated factor chart types.

    Cartesian projection:

    - `cartesian_chart(CartesianProductChart(...))` is applied factorwise.
    - Returns the same object if every factor is already Cartesian.
    - Otherwise returns a new `CartesianProductChart` with transformed factors and unchanged `factor_names`.

    Transformation behavior:

    - Inherits factorwise `pt_map` behavior from `AbstractCartesianProductChart` dispatches.

!!! info `SpaceTimeCT` and `spacetimect`

    Concrete flat-key spacetime product chart with Minkowski-oriented convention.

    `SpaceTimeCT` is a final `AbstractFlatCartesianProductChart` representing:
    $$
    \text{SpaceTimeCT(spatial\_chart)} \equiv \text{time1d} \times \text{spatial\_chart}
    $$
    with components `(ct, *spatial_components)`.

    API:

    - `spatial_chart`: spatial factor chart (default `cart3d`).
    - `c`: speed of light parameter (default `299792.458 km/s`).
    - `factors` is always `(time1d, spatial_chart)`.
    - `factor_names` is always `("time", "space")`.
    - `time_chart` is always `time1d` (not user-selectable).
    - `split_components(p)` returns `({"ct": ...}, {spatial keys...})`.
    - `merge_components(parts)` merges those two dicts directly.

    Component and dimension convention:

    - Overrides default product keys to use **flat** keys:
      - components: `("ct", *spatial_chart.components)`
      - dimensions: `("length", *spatial_chart.coord_dimensions)`
    - `ct` is the chart-level time coordinate key used by this representation.

    Cartesian projection:

    - `cartesian_chart(SpaceTimeCT(...))` maps only the spatial factor to its Cartesian chart.
    - Returns the same object when spatial factor is already Cartesian; otherwise returns a replaced `SpaceTimeCT`.

    Transformation behavior:

    - Uses the generic product-chart factorwise `pt_map` dispatch.
    - In practice, this preserves `ct` while transforming only spatial components when changing the spatial chart.

### Two-Sphere Charts

!!! info `AbstractSphericalTwoSphere`

    Abstract base class for intrinsic 2-sphere charts.

    - `AbstractSphericalTwoSphere` subclasses `Abstract2D` and represents charts on the curved unit sphere surface.
    - Components are purely angular (no radial component); these charts parameterize points on \(S^2\), not \(\mathbb{R}^2\).
    - There is **no global Cartesian 2D chart** for this manifold family.
    - `cartesian_chart(AbstractSphericalTwoSphere)` raises `NoGlobalCartesianChartError`.
    - For Euclidean embedding behavior, use a 3D embedding chart workflow (not intrinsic 2-sphere Cartesianization).

!!! info `SphericalTwoSphere` and `sph2`

    Physics-convention intrinsic 2-sphere chart and its pre-defined instance.

    - `SphericalTwoSphere` is the final chart type with components `("theta", "phi")`, dimensions `("angle", "angle")`.
    - Convention: `theta` is polar/colatitude \([0,\pi]\), `phi` is azimuth.
    - `check_data` enforces valid polar range for `theta` (`0°..180°`).
    - `sph2` is the pre-defined `SphericalTwoSphere()` instance.
    - Registered direct transitions:
      - `SphericalTwoSphere <-> LonLatSphericalTwoSphere`
      - `SphericalTwoSphere <-> LonCosLatSphericalTwoSphere`
      - `SphericalTwoSphere <-> MathSphericalTwoSphere`

!!! info `LonLatSphericalTwoSphere` and `lonlat_sph2`

    Longitude-latitude intrinsic 2-sphere chart and its pre-defined instance.

    - `LonLatSphericalTwoSphere` is the final chart type with components `("lon", "lat")`, dimensions `("angle", "angle")`.
    - `lat` is constrained to \([-90^\circ, 90^\circ]\); `lon` is azimuth/longitude.
    - `check_data` enforces latitude range.
    - `lonlat_sph2` is the pre-defined `LonLatSphericalTwoSphere()` instance.
    - Relation to physics spherical chart:
      - `lat = pi/2 - theta`
      - `lon = phi`
    - Registered direct transitions with `SphericalTwoSphere` implement these formulas.

!!! info `LonCosLatSphericalTwoSphere` and `loncoslat_sph2`

    Longitude-cos(latitude) intrinsic 2-sphere chart and its pre-defined instance.

    - `LonCosLatSphericalTwoSphere` is the final chart type with components `("lon_coslat", "lat")`, dimensions `("angle", "angle")`.
    - `lat` is constrained to \([-90^\circ, 90^\circ]\).
    - `lon_coslat` stores longitude weighted by `cos(lat)` to improve behavior near poles.
    - `check_data` enforces latitude range.
    - `loncoslat_sph2` is the pre-defined `LonCosLatSphericalTwoSphere()` instance.
    - Registered direct transitions with `SphericalTwoSphere`:
      - forward: `lat = pi/2 - theta`, `lon_coslat = phi * cos(lat)`
      - inverse: `theta = pi/2 - lat`, `phi = lon_coslat / cos(lat)`

!!! info `MathSphericalTwoSphere` and `math_sph2`

    Math-convention intrinsic 2-sphere chart and its pre-defined instance.

    - `MathSphericalTwoSphere` is the final chart type with components `("theta", "phi")`, dimensions `("angle", "angle")`.
    - Convention differs from physics spherical chart:
      - `theta` is azimuth
      - `phi` is polar angle
    - `check_data` enforces polar range for `phi` (`0°..180°`).
    - `math_sph2` is the pre-defined `MathSphericalTwoSphere()` instance.
    - Registered direct transitions with `SphericalTwoSphere` are component swaps:
      - physics -> math: `theta <- phi`, `phi <- theta`
      - math -> physics: `theta <- phi`, `phi <- theta`

### Poincare Charts

!!! info `PoincarePolar6D` and `poincarepolar6d`

    Concrete 6-D Poincare-polar chart and its pre-defined instance.

    - `PoincarePolar6D` is a final concrete chart type in the 6-D chart family.
    - Components (ordered):
      `("rho", "pp_phi", "z", "dt_rho", "dt_pp_phi", "dt_z")`.
    - Coordinate dimensions (ordered):
      `("length", "length / time**0.5", "length", "speed", "length / time**1.5", "speed")`.
    - `poincarepolar6d` is the pre-defined `PoincarePolar6D()` instance.
    - Transition behavior currently registered in-core:
      identity transform only (`pt_map(p, PoincarePolar6D, PoincarePolar6D) -> p`).
    - No dedicated Cartesian projection dispatch is currently defined for this chart family.

</br>

(software-spec-representations)=

## Representations

Building off transition maps and the other transformation laws we introduce a generalization over transformation laws that streamlines software implementation.

A `Representation` specifies _what kind of geometric object_ component data is meant to represent, independently of _which chart_ is used to write down the underlying coordinates. In `coordinax`, a representation is structured from three pieces:

1. a **geometry kind** ([`AbstractGeometry`](#software-spec-abstractgeometry)), which identifies the geometric type of the object (for example, a point, or tangent or cotangent object),
2. a **basis** ([`AbstractBasis`](#software-spec-abstractsbasis)), which identifies the basis in which components are expressed when such a choice is meaningful, and
3. a **semantic kind** ([`AbstractSemanticKind`](#software-spec-abstractsemantickind)), which identifies the physical or mathematical meaning attached to that geometric type. Formally, one may view a representation as a triple

$$
R = (K, B, S),
$$

where $K$ is the geometry kind, $B$ is the basis choice, and $S$ is the semantic kind.

A representation is therefore **not** the same thing as a chart: the chart determines the coordinate system, while the representation determines the transformation law and geometric interpretation of the data.

!!! example

    For **points**, the representation determines that the transformation law is a [transition map](#math-spec-transition-maps).

!!! info `Representation`

    ``Representation``.

    Arguments:

    - ``geom_kind``: ``AbstractGometry``
    - ``basis``: ``AbstractBasis``
    - ``semantic_kind``: ``AbstractSemanticKind``

    <!-- skip: next -->

    ```python
    @jax.tree_util.register_static
    @dataclass
    class Representation:
        geom_kind: AbstractGometry
        basis: AbstractBasis
        semantic_kind: AbstractSemanticKind
    ```

(software-spec-singletons)=

!!! info Pre-defined Representations

    `Representation` instances for standard use cases.

    |     Name     |   `geom_kind`  |    `basis`    | `semantic_kind` |
    |--------------|----------------|---------------|-----------------|
    | `point`      | [`point_geom`](#software-spec-point-geometry)   | [`no_basis`](#software-spec-no_basis)    |      [`loc`](#software-spec-location)      |

</br>

### Functional API

!!! info `cconvert`

    `cconvert` is the highest-level part of `coordinax`'s coordinate transformation machinery.
    It abstracts over all the specific transformation machinery:

    - [`coordinax.charts.pt_map`](#software-spec-pt_map)

    Consequently, it uses (/needs) more structure to distinguish between transformations.

    Registered Dispatches (with $f$ final and $i$ initial):

    - $(C_{f}, R_{f}, C_{i}, R_{i})$ -> $(C_{f}, K_{f}, R_{f}, C_{i}, K_{i}, R_{i})$ general redispatch to use the geometric type in the transformation selection.
    - $(C_{f}, \text{PointGeometry}, R_{f}, C_{i}, \text{PointGeometry}, R_{i})$ redispatch to use the transition map $\varphi_{C_{f}} \circ \varphi_{C_{i}}^{-1}$. $R_{i,f}$ are checked that [B=NoBasis()](#software-spec-no_basis), [S=Location()](#software-spec-location)

(software-spec-guess_rep)=

!!! info `guess_rep`

    Infer the full representation triple from lightweight structural information.

    Dispatches:

    - `Representation` -> identity.
    - `PointGeometry` -> returns `point`.
    - `u.AbstractDimension | u.AbstractQuantity | CDict` -> infer via `guess_geometry_kind`, then redispatch on geometry kind.
    - `(Any, AbstractChart)` -> infer via `guess_geometry_kind(obj, chart)`, then redispatch on geometry kind.

    Failure semantics:

    - Inherits all failure semantics from `guess_geometry_kind`.

</br>

### Geometric Kind

A **geometry kind** identifies the mathematical type of geometric object that component data represents. Importantly, the geometry kind is **independent of the coordinate chart** used to represent the components. The same geometric object may be written in any compatible chart.

Geometric objects arise from geometric structures associated with a manifold $M$. A geometry kind associates a set of coordinates with one such structure, and thereby specifies what sort of object the coordinates are describing. Each geometry kind determines the **coordinate transformation law** for components.

- **Points**: a point is an element $$p \in M .$$ If $q_i$ and $q_f$ are two coordinate descriptions of the same point, then the transformation law is $$
q_f = \tau(q_i) .$$ An important example is the **point geometry**.

(software-spec-abstractgeometry)=

!!! info `AbstractGeometry`

    Abstract base class for geometric kind. `AbstractGeometry` identifies the geometric type of represented data, independent of chart and basis. It answers: "what geometric object do these components represent?" (for example, a point). In the representation triple $R = (K, B, S)$, `AbstractGeometry` is the `K` component.

    Transformation-law role:

    - Geometric kind determines the abstract coordinate-change law for
        components.
    - Point-kind data transforms by chart transition maps.
    - Future tangent/cotangent kinds transform by Jacobian pushforward / pullback laws, respectively.

    Distinctions:

    - Not a chart: charts define coordinate systems, component names, and domains.
    - Not a semantic kind: semantics encode interpretation (for example, location, velocity) within a fixed geometric kind.
    - Not a basis: basis selection is tracked separately and may be trivial for affine objects.

    Notes:

    - `AbstractGeometry` is a static dispatch object category (no runtime numerical payload).
    - Concrete subclasses should represent immutable geometric categories.

(software-spec-guess_geometry_kind)=

!!! info `guess_geometry_kind`

    Infer geometry kind from lightweight structural information.

    Dispatches:

    - `AbstractGeometry` -> identity.
    - `u.AbstractDimension` -> look up in `DIM_TO_GEOM_MAP` (`"length"`, `"angle"` -> `PointGeometry`).
    - `u.AbstractQuantity` -> extract dimension via `unxt.dimension_of`, then redispatch.
    - `CDict` -> collect dimensions; discard `"angle"` when mixed; infer from remainder.
    - `(CDict, AbstractChart)` -> validate key schema via `check_data`, then redispatch on `CDict`.
    - `(CDict, ProlateSpheroidal3D)` -> special-case: `{area, angle}` -> `PointGeometry`.

    Failure semantics:

    - Empty mapping -> `ValueError`.
    - Multiple non-angle dimensions after discarding -> `ValueError`.
    - Unregistered dimension -> `ValueError`.
    - Prolate spheroidal: dimensions not `{area, angle}` -> `ValueError`.

!!! info `PointGeometry` and `point_geom`

    Concrete geometric kind for manifold points, and its canonical instance.

    - `PointGeometry` is the final concrete subclass of `AbstractGeometry` for point-like data.
    - It encodes that components represent a point $p \in M$ (an affine object), not a vector in a tangent/cotangent space.
    - Point coordinates therefore transform by the ordinary chart transition map (`pt_map` / `pt_map` point behavior), with the represented geometric object unchanged.

    Affine semantics:

    - Point data is location data: points do not add as vectors.
    - Any vector-space operations require a separate geometric kind (for example tangent/cotangent kinds), not `PointGeometry`.

    API instance:

    - `point_geom` is the pre-defined canonical `PointGeometry()` instance used by the default point representation `point = Representation(point_geom, no_basis, loc)`.

</br>

### Basis

Many geometric objects live in vector spaces. In such spaces, component values depend on the choice of basis.

A **basis** $B$ for a vector space $V$ is a linearly independent set

$$
{ e_a }_{a=1}^{\dim V}
$$

such that any vector $v \in V$ can be written uniquely as

$$
v = v^a e_a .
$$

The coefficients $v^a$ are the **components of the vector in the basis** $B$.

However, not all geometric objects require a basis specification.

- Points are affine objects and do not belong to a vector space. Their coordinates are chart values, not vector components.

(software-spec-abstractsbasis)=

!!! info `AbstractBasis`

    Abstract base class for basis kind.

    - `AbstractBasis` identifies the component basis in which represented data is written, when basis choice is meaningful.
    - In the representation triple $R = (K, B, S)$, `AbstractBasis` is the `B` component.
    - It answers: "in what basis are these components expressed?"

    Role in representation:

    - Basis is orthogonal to chart, geometry kind, and semantic kind.
    - Chart defines coordinates and coordinate domains.
    - Geometry kind defines the geometric object and transformation law class.
    - Semantic kind defines interpretation (for example location or velocity).

    Basis-law role:

    - For affine point data, basis choice is trivial (`NoBasis`).

    Notes:

    - `AbstractBasis` is a static dispatch object category.
    - Concrete subclasses should represent immutable basis categories.

(software-spec-guess_basis_kind)=

!!! info `guess_basis_kind`

    Infer basis kind from lightweight structural information.

    Dispatches:

    - `AbstractBasis` -> identity.
    - `u.AbstractDimension` -> look up in `DIM_TO_BASIS_MAP` (`"length"`, `"angle"` -> `NoBasis`).
    - `u.AbstractQuantity` -> extract dimension via `unxt.dimension_of`, then redispatch.
    - `CDict` -> collect dimensions; infer from remainder.

    Failure semantics:

    - Empty mapping -> `ValueError`.
    - Unregistered dimension -> `ValueError`.

(software-spec-no_basis)=

!!! info `NoBasis` and `no_basis`

    Concrete no-basis kind, and its canonical instance.

    - `NoBasis` is the final concrete subclass of `AbstractBasis` used when represented data is not expressed in a basis-dependent linear space.
    - Canonical use: point data (`PointGeometry`), where coordinates describe locations on a manifold rather than vector components in $T_pM$.

    Semantics:

    - `NoBasis` does **not** mean "no coordinates".
    - It means basis choice is not part of the representation semantics for the object kind.
    - Coordinate changes for point data remain chart transition maps, not basis changes.

    API instance:

    - `no_basis` is the pre-defined canonical `NoBasis()` instance.
    - It is used in the default point representation `point = Representation(point_geom, no_basis, loc)`.

</br>

### Semantic Kind

A **semantic kind** specifies the **interpretation** attached to a geometric object.

While the geometry kind determines the mathematical space of the object, the semantic kind distinguishes _how that object is interpreted physically_.

Examples include:

- **Location**: a point interpreted as the position of a particle or object.

Semantic kinds do **not** change coordinate transformation laws.

Separating semantics from geometry provides two advantages:

1. Correct transformation laws -- transformation behavior depends only on geometry kind and basis, not on semantics.
2. Clear interpretation -- different semantic kinds distinguish objects that share the same mathematical type but represent different physical quantities.

(software-spec-abstractsemantickind)=

!!! info `AbstractSemanticKind`

    Abstract base class for semantic kind.

    - `AbstractSemanticKind` identifies the interpretation attached to represented data.
    - In the representation triple $R = (K, B, S)$, `AbstractSemanticKind` is the `S` component.
    - It answers: "what does this represented object mean?"

    Role in representation:

    - Semantic kind is orthogonal to chart, geometry kind, and basis.
    - Chart defines coordinate system and component domains.
    - Geometry kind defines the geometric object and transformation-law class.
    - Basis defines component basis when basis choice is meaningful.

    Semantic-law role:

    - Semantic kind refines interpretation within a fixed geometric kind and basis without changing coordinate transformation laws.

    Notes:

    - `AbstractSemanticKind` is a static dispatch object category (no runtime numerical payload).
    - Concrete subclasses should represent immutable semantic categories.

(software-spec-guess_semantic_kind)=

!!! info `guess_semantic_kind`

    Infer semantic kind from lightweight structural information.

    Dispatches:

    - `AbstractSemanticKind` -> identity.
    - `u.AbstractDimension` -> look up in `DIM_TO_SEMANTICS_MAP` (`"length"`, `"angle"` -> `Location`).
    - `u.AbstractQuantity` -> extract dimension via `unxt.dimension_of`, then redispatch.
    - `CDict` -> collect dimensions; discard `"angle"` when mixed; infer from remainder.

    Failure semantics:

    - Empty mapping -> `ValueError`.
    - Multiple non-angle dimensions after discarding -> `ValueError`.
    - Unregistered dimension -> `ValueError`.

(software-spec-location)=

!!! info `Location` and `loc`

    Concrete location semantic kind, and its canonical instance.

    - `Location` is the final concrete subclass of `AbstractSemanticKind` used to denote "where" data: coordinates interpreted as position on a manifold.
    - Canonical pairing in current `coordinax`: point geometry with no basis, i.e. `(PointGeometry, NoBasis, Location)`.

    Semantics:

    - `Location` records interpretation, not chart choice.
    - It does not alter the underlying point transformation law: point coordinates still transform by chart transition maps.
    - It distinguishes location-like point data from other possible semantics introduced for non-point geometric kinds.

    API instance:

    - `loc` is the pre-defined canonical `Location()` instance.
    - It is used in the default point representation `point = Representation(point_geom, no_basis, loc)`.


</br>

(software-spec-vectors)=

## Vectors

!!! info `AbstractVector`

    Methods \& Properties:

    - ``from_()``: multiple dispatch constructor of vector-like objects from arguments.
    - ``uconvert``: convert the vector to the given units.
      - ``(V, *args, **kwargs) -> uconvert(*args, V, **kwargs)`` redispatch
      - ``(V, u.AbstractUnitSystem) -> uconvert(u.AbstractUnitSystem, V)`` redispatch
    - ``__array_namespace__``: the array API namespace -- `quax.numpy`. This delegates to `quax` primitives.
    - ``__eq__``: equality check, based on type equality and `quax` equality primitive.
    - ``copy()``: call `dataclass.replace`.
    - ``flatten()``: flatten the vector.
    - ``ravel()``: return a flattened vector.
    - ``reshape(*shape)``: return a reshaped vector
    - ``round(decimals)``: return a rounded vector.
    - ``to_device(device)``: move the vector to a new device
    - ``__repr__()``: string representation through `wadler_lindig`.
    - ``__str__()``: string representation through `wadler_lindig`.
    - ``is_like()``: check if the object is a `AbstractVector` object.

    Abstract Methods \& Properties:

    - ``shape``: the shape of the vector. Abstract method.
    - ``__getitem__(slice)``: slice the vector.
    - ``astype(dtype, **kw)`` : cast the vector to a new dtype.

    Not Supported:

    - ``materialise``: for materialising the vector for `quax`.
    - ``__complex__``
    - ``__float__``
    - ``__index__``
    - ``__int__``
    - ``__setitem__`` : vectors are immutable.
    - ``__hash__()``: hash the vector by its field items. In general this raises an error.

!!! info `Point`

    Arguments:

    - ``data``: the data for each component.
    - ``chart``: the chart of the vector, e.g. ``cxc.cart3d``.
    - ``rep``: the `coordinax.representations.Representation`, e.g. `cxr.point`.

    Methods \& Properties:

    - ``__getitem__()``
    - ``__pdoc__()``
    - ``cconvert()``
    - ``aval()``
    - ``shape``
    - ``norm()``

    `from_` Constructor Dispatches:

    - ``(Point,) -> Point``
    - ``(dict,C,R) -> Point``
    - ``(dict,C) -> (dict,C,R) -> Point``
    - ``(dict,) -> (dict,C,R) -> Point``
    - ``(Q,C,R) -> (dict,C,R) -> Point``
    - ``(Q,C) -> (dict,C,R) -> Point``
    - ``(Q,) -> (dict,C) -> (dict,C,R) -> Point``
    - ``(Q,R) -> (dict,C,R) -> Point``
    - ``(Array,unit) -> (Q,) -> ... -> Point``
    - ``(Array,unit,C) -> ... -> Point``
    - ``(Array,unit,C,R) -> ... -> Point``

!!! info `ToUnitsOptions`

    Used for `unxt.uconvert` dispatches.
