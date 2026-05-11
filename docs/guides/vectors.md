# Working With Vectors

This guide provides a conceptual introduction to coordinax vectors and practical patterns for working with them. For API reference, see [the vector module reference](../api/vectors.md).

## Motivation: Why A Separate Vector Class?

In pure NumPy or JAX, coordinate data is just arrays. But astronomy and geometry demand more:

1. **Coordinate systems vary**: the same point is `(x, y, z)` in Cartesian but `(r, θ, φ)` in spherical. Which is it?
2. **Units matter**: is `1.0` in meters, parsecs, or degrees? Silent unit confusion causes disasters.
3. **Transformation laws differ**: point coordinates change by the chart transition; vector fields transform by the Jacobian. These rules cannot be implicit.
4. **Type safety**: mixing spherical and Cartesian accidentally should be impossible, not silently wrong.

Coordinax `Point` solves this by **making all three explicit**: chart (system), data (values), and representation (meaning).

## From Charts to Vectors

If you have not yet read [Working With Charts](./charts.md), do so first. Charts define coordinate systems; vectors express data in those systems.

**Chart's job**: define component names and their physical dimensions.

```python
import coordinax.charts as cxc

cart = cxc.cart3d  # Components: x, y, z (all length)
sph = cxc.sph3d  # Components: r (length), theta (angle), phi (angle)
```

**Point's job**: store data _in a specific chart_ with an explicit transformation law.

```python
import coordinax.main as cx
import unxt as u

# A point in Cartesian space
p_cart = cx.Point.from_(
    {"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}, cxc.cart3d, cx.point
)

# Same point in spherical (after transformation)
p_sph = cx.cconvert(p_cart, cxc.sph3d)
print(p_sph.data)  # {"r": ..., "theta": ..., "phi": ...}
```

The chart and representation together ensure that:

- You know what the numbers mean
- Transformations apply the correct mathematical laws
- Mixing incompatible data is impossible

## Constructor Patterns

Vectors support flexible construction via `from_()`:

### 1. From Array + Unit (Simplest)

```python
import coordinax.main as cx

# Shape infers chart (3 → cart3d, 2 → cart2d, etc.)
v = cx.Point.from_([1, 2, 3], "m")
```

The chart is inferred from the array shape:

- Shape `(3,)` → `cart3d`
- Shape `(2,)` → `cart2d`
- Shape `(1,)` → `cart1d`
- Shape `()` → `cart0d`

### 2. With Explicit Chart

```python
import coordinax.charts as cxc

# Override inferred chart
v = cx.Point.from_([1, 2, 3], "m", cxc.sph3d)
```

### 3. From Quantity

```python
import unxt as u

# Quantity already has units
q = u.Q([1, 2, 3], "m")
v = cx.Point.from_(q)
```

### 4. From Component Dictionary

```python
import coordinax.charts as cxc
import unxt as u

# Most explicit: name each component
v = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}, cxc.cart3d)
```

This is the **most defensive** pattern: each component is named and units are explicit.

### 5. Passthrough (Already a Point)

```python
v1 = cx.Point.from_([1, 2, 3], "m")
v2 = cx.Point.from_(v1)  # Returns v1 unchanged
```

## Coordinate Transformations

Use `cconvert()` to change coordinate systems while preserving the geometric point:

```python
import coordinax.main as cx
import coordinax.charts as cxc
import unxt as u

# Start in Cartesian
v_cart = cx.Point.from_([1, 2, 3], "m")

# Convert to spherical
v_sph = cx.cconvert(v_cart, cxc.sph3d)

# Data is now in spherical coordinates
print(v_sph.data)
# {"r": Array(...), "theta": Array(...), "phi": Array(...)}

# And back
v_cart_again = cx.cconvert(v_sph, cxc.cart3d)
```

**Key point**: `cconvert` preserves the geometric meaning (same point), not the numbers. The chart _changes_; the point _stays_.

## Unit Conversion

Separate from chart conversion is unit conversion:

```python
import coordinax.main as cx
import unxt as u

v = cx.Point.from_([1000, 2000, 3000], "m")

# Convert to km (same chart, different units)
v_km = u.uconvert({"x": "km", "y": "km", "z": "km"}, v)
# Data is now [1, 2, 3] but in km

# Can also convert per-component
v_mixed = u.uconvert({"x": "km", "y": "km", "z": "m"}, v)
```

## Coordinate Objects

`Vector` answers: "what are the component values, chart, and representation?"

`Coordinate` answers: "what are the vector components **and** in which reference frame are they described?"

A coordinate is therefore:

- vector data (component values)
- chart (coordinate system)
- representation (transformation law)
- frame (observer)

### Choosing The Right Operation

Use this decision table when transforming coordinate data:

| Goal | API | What Changes | What Stays Invariant |
| --- | --- | --- | --- |
| Change observer/reference frame | `coord.to_frame(to_frame)` | `frame` (and usually component values) | represented geometric point |
| Change coordinate system/chart | `coord.cconvert(to_chart)` | `chart` (and usually component values) | represented geometric point, `frame` |
| Change both | chain both operations | both frame and chart | represented geometric point |

### Constructing Coordinates

`Coordinate.from_()` supports the same flexible input styles as `Vector.from_()`, with an additional frame argument.

```python
import coordinax.main as cx
import coordinax.frames as cxf

# 1) Array + unit + frame (common high-level pattern)
coord1 = cx.Point.from_([1, 2, 3], "kpc", cxf.alice)

# 2) Vector + frame
vec = cx.Point.from_([1, 2, 3], "kpc")
coord2 = cx.Point.from_(vec, cxf.alice)

# 3) No explicit frame -> NoFrame()
coord3 = cx.Point.from_([1, 2, 3], "kpc")
print(coord3.frame)  # NoFrame()

# 4) Passthrough
coord4 = cx.Point.from_(coord1)
```

For explicit control, you can also pass chart/representation/manifold through constructor dispatches that mirror `Vector.from_()`.

### Frame Transformations Of Coordinate Data

Use `to_frame()` to apply the frame-transition operator associated with the new observer. Under the active convention, this moves the represented point data into the target frame.

```python
import coordinax.main as cx
import coordinax.frames as cxf

coord_a = cx.Point.from_([1, 2, 3], "m", cxf.alice)
coord_b = coord_a.to_frame(cxf.alex)

print(coord_a.frame)  # Alice()
print(coord_b.frame)  # Alex()
```

Identity frame changes are cheap no-ops:

```python
same = coord_a.to_frame(cxf.alice)
print(same is coord_a)  # True
```

For evolving transforms, `to_frame()` accepts an optional evolution parameter `t` (a quantity):

```python
import unxt as u

coord_t = coord_a.to_frame(cxf.alex, t=u.Q(1.0, "s"))
```

You can also apply frame operators directly:

```python
op = cxf.frame_transition(cxf.alice, cxf.alex)
coord_b2 = op(coord_a)
```

### Chart Transformations Of Coordinate Data

Use `cconvert()` to change chart representation while preserving frame.

```python
import coordinax.charts as cxc

coord_cart = cx.Point.from_([1, 2, 3], "m", cxf.alice)
coord_sph = coord_cart.cconvert(cxc.sph3d)

print(coord_cart.chart)  # Cart3D(M=Rn(3))
print(coord_sph.chart)  # Spherical3D()

print(coord_cart.frame)  # Alice()
print(coord_sph.frame)  # Alice()  (unchanged)
```

Round-tripping charts preserves the represented point semantics:

```python
coord_cart_again = coord_sph.cconvert(cxc.cart3d)
```

### Combined Frame + Chart Pipelines

Real workflows often require both transformations.

```python
import coordinax.charts as cxc
import coordinax.frames as cxf
import coordinax.main as cx

coord = cx.Point.from_([10, 5, 2], "kpc", cxf.alice)

# Pipeline A: frame first, then chart
out_a = coord.to_frame(cxf.alex).cconvert(cxc.sph3d)

# Pipeline B: chart first, then frame
out_b = coord.cconvert(cxc.sph3d).to_frame(cxf.alex)

print(out_a.frame, out_a.chart)  # Alex(), Spherical3D()
print(out_b.frame, out_b.chart)  # Alex(), Spherical3D()
```

When reading a pipeline, track invariants explicitly:

1. Frame operation changes observer metadata.
2. Chart operation changes component schema/coordinate system.
3. Both preserve the represented geometric point semantics.

### Batch And JAX Patterns For Coordinates

Use scalar-first functions, then batch with `vmap`.

```python
def to_alex_spherical(c):
    return c.to_frame(cxf.alex).cconvert(cxc.sph3d)


coords = [cx.Point.from_([i, i + 1, i + 2], "m", cxf.alice) for i in range(5)]
coords_out = [to_alex_spherical(c) for c in coords]
```

### Astronomy-Oriented Pattern (Optional)

If astronomy frames are available, Coordinate workflows are identical:

```python
# Optional: requires astro frame package/config in your environment
# import coordinax.astro as cxastro
# coord_icrs = cx.Point.from_([1, 2, 3], "kpc", cxastro.ICRS())
# coord_gc = coord_icrs.to_frame(cxastro.Galactocentric())
# coord_gc_sph = coord_gc.cconvert(cxc.sph3d)
```

This is the same pattern: first choose observer frame semantics, then choose the most useful chart for the analysis step.

## Architecture & Design Philosophy

The coordinax `Point` is built on a **"data + chart + representation" triple**. This design prevents silent coordinate errors and makes transformations explicit.

### Why Combine Data + Chart + Representation?

In raw NumPy/JAX, coordinates are just arrays. This creates problems:

```python
import jax.numpy as jnp

# BAD: What does this mean?
x = jnp.array([1, 2, 3])
# Is it Cartesian (x, y, z) or spherical (r, θ, φ)?
# Are the units meters or degrees?
# Should transformations apply the chart transition or Jacobian?
```

Coordinax solves this by making all three explicit:

```python
# GOOD: Crystal clear
v = cx.Point.from_({"x": u.Q(1, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")}, cxc.cart3d)
# → Chart says: components are (x, y, z) in meters
```

**Benefits:**

1. **Type Safety**: mixing `cart3d` and `sph3d` points is a type error, not silent garbage
2. **Semantic Clarity**: each number has explicit meaning (system, units, transformation law)
3. **Automatic Correctness**: transformations apply the right rules by construction

### Immutability by Design

Vectors are immutable. To modify a vector, use `dataclass.replace()`:

```python
import coordinax.main as cx
from dataclasses import replace

v = cx.Point.from_([1, 2, 3], "m")

# Create a new point with modified data
v_modified = replace(
    v,
    data={"x": u.Q(10, "m"), "y": u.Q(2, "m"), "z": u.Q(3, "m")},
)

# Original unchanged
assert v.data["x"] != v_modified.data["x"]
```

Immutability enables:

- **JAX safety**: no hidden mutations during transformations
- **Functional composition**: pure functions chain reliably
- **Caching**: same input always produces same result

## Arithmetic & Operations

Vectors support Quax-style arithmetic:

```python
import coordinax.main as cx

v1 = cx.Point.from_([1, 2, 3], "m")
v2 = cx.Point.from_([0.5, 0.5, 0.5], "m")

# Example operations (availability may depend on branch state):
# v_sum = v1 + v2
# v_diff = v1 - v2
# v_scaled = v1 * 2
# dist = v_diff.norm()
```

**Implementation**: Operators use Quax dispatch on JAX primitives (`lax.add_p`, `lax.mul_p`, etc.), making them fully JAX-compatible. See [JAX Integration](#jax-integration--scaling) below.

## Representations: Point vs Tangent (Future)

Currently, vectors support the **point representation** (coordinates transform by chart change only).

Future work will add:

- **Tangent vectors**: transform by the Jacobian of the coordinate change
- **Covectors**: transform inverse-transpose
- **Densities**: transform with the determinant of the Jacobian

For now, all vectors are points. The `rep` parameter is present for forward compatibility.

## JAX Integration & Scaling

Vectors are designed as JAX PyTrees and integrate seamlessly with JAX transformations. This section covers the implementation details and practical patterns.

### Scalar-First Design

Vectors are designed to operate on **scalar data** (individual points, not batches):

```python
import coordinax.main as cx

# Scalar operation
v = cx.Point.from_([1, 2, 3], "m")
# Components are leaves: {"x": 1.0, "y": 2.0, "z": 3.0}

# Don't design for: v = cx.Point.from_(np.array([[1, 2, 3], [4, 5, 6]]), "m")
# (This would make components shaped, not scalar)
```

Instead, use **vmap** for batching:

### PyTree Structure & Flattening

Vectors are registered as JAX PyTrees via Equinox. This means:

```python
import jax
import coordinax.main as cx

v = cx.Point.from_([1, 2, 3], "m")

# JAX sees Point as a pytree with:
# - Children (leaves): the `data` dictionary components
# - Metadata: chart, representation, other non-array attributes

flat, treedef = jax.tree_util.tree_flatten(v)
# flat = [1.0, 2.0, 3.0]  (the actual arrays)
# treedef = <PyTreeDef with structure and metadata>

# Reconstruct
v_reconstructed = jax.tree_util.tree_unflatten(treedef, flat)
```

**Why this matters**:

- `jit`, `vmap`, `grad` automatically traverse the tree correctly
- Chart and representation metadata are preserved (not treated as data)
- Transformations view vectors as a whole, not element-by-element

### Quax Dispatch for Operators

Vector operators (`+`, `-`, `*`, etc.) are implemented via **Quax multiple dispatch** on JAX primitives, not as Python magic methods. This enables JAX transformations to handle them correctly:

```python
# Point implements quax.ArrayValue interface
# Operators are dispatched on lax primitives, e.g.:
# @quax.register(lax.add_p)
# def add_points(v1: Point, v2: Point, /) -> Point:
#     new_data = {k: v1.data[k] + v2.data[k] for k in v1.data}
#     return Point(new_data, chart=v1.chart)
```

**Why Quax (not Python magic methods)**:

- `jnp.add(v1, v2)` decomposes to JAX primitives at JIT time
- Static dispatch before JAX tracing preserves semantics
- Works with `vmap`, `grad`, `jit` without modification

### Vectorization via Vmap

```python
import coordinax.main as cx


# Scalar operation: convert one vector to spherical
def transform_one(v):
    return cx.cconvert(v, cxc.sph3d)


# Batch operation (doctest-safe example)
many_vectors = [cx.Point.from_([i, i + 1, i + 2], "m") for i in range(5)]
many_spherical = [transform_one(v) for v in many_vectors]
```

Vectors are JAX PyTrees, so they work with `jit`:

```python
# Example sketch (availability depends on branch state):
# import jax
#
# @jax.jit
# def compute_distances(v1, v2):
#     diff = v1 - v2
#     return diff.norm()
#
# result = compute_distances(vec1, vec2)
```

### Differentiation via Grad

Compute gradients through vector operations:

```python
# Example sketch (availability depends on branch state):
# @jax.grad
# def loss(v):
#     # Example: minimize norm
#     return v.norm() ** 2
#
# grad_v = loss(vec)
```

## Common Pitfalls

### 1. Silent Chart Mismatch (Prevented!)

```python
v_cart = cx.Point.from_([1, 2, 3], "m", cxc.cart3d)
v_sph = cx.Point.from_([1, 2, 3], "m", cxc.sph3d)

# This is caught by the type system (different charts)
# v_sum = v_cart + v_sph  # ERROR: incompatible charts
```

### 2. Mutating Vectors (Not Allowed)

```python
v = cx.Point.from_([1, 2, 3], "m")

# This is an error (immutable):
# v.data["x"] = 10*u.m

# Instead, use replace:
from dataclasses import replace

v = replace(v, data=v.data)
```

### 3. forgetting Units

```python
# WRONG: units are implicit
# v = cx.Point.from_([1, 2, 3])  # ERROR: no unit specified

# RIGHT: always specify units
v = cx.Point.from_([1, 2, 3], "m")
```

### 4. Shape Inference Ambiguity

```python
# Shape (1,) could be 1D Cartesian or a scalar
# Prefer explicit:
v = cx.Point.from_([1.5], "m", cxc.cart1d)
```

## Summary: Workflow Pattern

1. **Construct** from array, quantity, or dict
2. **Inspect** data (print, check shape, extract components)
3. **Convert** between charts if needed (cconvert)
4. **Transform** units if needed (uconvert)
5. **Operate** (arithmetic, norm, reshape)
6. **Scale** with vmap/jit for batching and compilation
7. **Differentiate** with grad if needed

This workflow keeps coordinates clear, transformations correct, and code efficient.
