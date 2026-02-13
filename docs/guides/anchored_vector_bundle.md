# Anchored Vector Bundle

A `PointedVector` provides an ergonomic container for working with collections
of vectors anchored at a common base point, automatically handling the
coordinate transformation dependencies required for tangent vectors.

## Motivation

In differential geometry, tangent vectors (velocities, accelerations,
displacements) require a base point for coordinate transformations between
curvilinear systems. Manually tracking these dependencies leads to verbose code:

```
import coordinax.charts as cxc
from coordinax.objs import Vector

# Manual approach - verbose and error-prone
base = Vector.from_([1, 2, 3], "km")
vel = Vector.from_([10, 20, 30], "km/s")

# Convert to spherical - must manually provide base
base_sph = base.vconvert(cxc.sph3d)
at_for_vel = base.vconvert(vel.chart)  # Ensure base matches vel's rep
vel_sph = vel.vconvert(cxc.sph3d, at_for_vel)
```

`PointedVector` automates this bookkeeping:

```
import coordinax.objs as PointedVector

# Bundle approach - concise and safe
bundle = PointedVector(base=base, velocity=vel)
sph_bundle = bundle.vconvert(cxc.sph3d)  # Handles at= automatically
```

## Mathematical Foundation

### Vector Bundle over a Point

An anchored vector bundle represents a **vector bundle over a single base
point** $q \in M$ on a manifold $M$:

$$
E_q = \{q\} \cup \bigsqcup_{i} F_i|_q
$$

where:

- $q$ is the **base point** (role `Point`)
- $F_i|_q$ are **fibres** at $q$ (typically $T_q M$, the tangent space)
- Common fibre types: `PhysDisp`, `PhysVel`, `PhysAcc`

###Transformation Rules

When converting coordinates from representation $R$ to $S$:

**1. Base Point (Position)**

The base transforms via a coordinate chart map:

$$
q_S = f(q_R)
$$

This is a **point transformation** - no `at=` parameter needed.

**2. Fibre Vectors (Tangent Vectors)**

Tangent vectors transform via the **pushforward** (differential/Jacobian) at the
base point:

$$
v_S = \mathrm{d}f_{q_R}(v_R)
$$

This requires evaluating the transformation **at the base point in the source
representation**.

### Representation Matching

Critical detail: When converting a fibre vector $v$ in representation $R_v$ to
$S_v$, the base point must be provided in $R_v$ (not necessarily the bundle's
current base representation):

$$
\text{at} = q_{\text{bundle}}.\text{vconvert}(R_v) \quad \text{(position conversion)}
$$

`PointedVector.vconvert()` handles this automatically.

## Core Concepts

### Base vs. Fields

```
import coordinax as cx
import unxt as u

base = Vector.from_([1, 2, 3], "m")
vel = Vector.from_([4, 5, 6], "m/s")
acc = Vector.from_([0.1, 0.2, 0.3], "m/s^2")

bundle = cx.PointedVector(
    base=base,  # Position (role: Pos)
    velocity=vel,  # Tangent vector (role: Vel)
    acceleration=acc,  # Tangent vector (role: Acc)
)

# Access
bundle.base  # or bundle.q
bundle["velocity"]
bundle["acceleration"]
```

**Requirements:**

- `base` must have role `PhysDisp`
- Fields must NOT have role `PhysDisp` (enforced at construction)
- All vectors must have compatible (broadcastable) shapes

### Role-Based Transformation

| Role       | Transformation                       | Requires `at=`?       | Example               |
| ---------- | ------------------------------------ | --------------------- | --------------------- |
| `Point`    | Position map $f(q)$                  | No                    | Base point conversion |
| `PhysDisp` | Tangent transform $\mathrm{d}f_q(v)$ | Yes (for curvilinear) | Displacement vector   |
| `PhysVel`  | Tangent transform $\mathrm{d}f_q(v)$ | Yes (for curvilinear) | Velocity vector       |
| `PhysAcc`  | Tangent transform $\mathrm{d}f_q(v)$ | Yes (for curvilinear) | Acceleration vector   |

## Usage Examples

### Basic Conversion

```
import coordinax as cx

# Create bundle in Cartesian coordinates
base = Vector.from_([1, 1, 1], "m")
vel = Vector.from_([10, 10, 10], "m/s")
bundle = cx.PointedVector(base=base, velocity=vel)

# Convert entire bundle to spherical
sph_bundle = bundle.vconvert(cxc.sph3d)

# Both base and velocity are now in spherical coordinates
print(type(sph_bundle.base.chart))  # Spherical3D
print(type(sph_bundle["velocity"].chart))  # Spherical3D
```

### Mixed Target Representations

```
# Convert base to spherical, velocity to cylindrical
mixed_bundle = bundle.vconvert(
    cxc.sph3d, field_charts={"velocity": cxc.cyl3d}
)

print(type(mixed_bundle.base.chart))  # Spherical3D
print(type(mixed_bundle["velocity"].chart))  # Cylindrical3D
```

### Physical Vector Components

**Important**: Tangent vectors store _physical_ components in orthonormal frames
with uniform dimensions, NOT coordinate differentials.

```
# In cylindrical coordinates (ρ, φ, z):

# ✓ Correct: Physical velocity components (all m/s)
v_physical = {"rho": u.Q(1.0, "m/s"), "phi": u.Q(2.0, "m/s"), "z": u.Q(0.5, "m/s")}
# phi component is the tangential velocity (arc length / time)

# ✗ Incorrect: Mixed units [m/s, rad/s, m/s]
v_mixed = {"rho": u.Q(1.0, "m/s"), "phi": u.Q(0.1, "rad/s"), "z": u.Q(0.5, "m/s")}
# This would be coordinate differentials, not physical vectors
```

The bundle automatically uses `coordinax.transforms.physical_tangent_transform`
for fibre conversions, which expects homogeneous physical components.

### Batched Bundles

```
import jax.numpy as jnp

# Create batch of positions (2 points)
bases = Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "kpc")

# Single velocity (broadcasts to all bases)
vel = Vector.from_([10, 20, 30], "km/s")

bundle = cx.PointedVector(base=bases, velocity=vel)
print(bundle.shape)  # (2,)

# Index to get sub-vectors
bundle[0]  # First point-velocity pair
bundle[1]  # Second point-velocity pair
```

### Construction from Dictionaries

```
data = {
    "base": u.Q([1, 2, 3], "km"),
    "velocity": u.Q([4, 5, 6], "km/s"),
    "acceleration": u.Q([0.1, 0.2, 0.3], "km/s^2"),
}

bundle = cx.PointedVector.from_(data)
```

Or with explicit base:

```
base = Vector.from_([1, 2, 3], "km")
fields = {
    "velocity": u.Q([4, 5, 6], "km/s"),
    "momentum": u.Q([40, 50, 60], "kg*m/s"),  # Any tangent-like vector
}

bundle = cx.PointedVector.from_(fields, base=base)
```

## JAX Compatibility

### JIT Compilation

Bundles work with `jax.jit`, but representations must be static:

```
import jax


@jax.jit
def process_bundle(bundle):
    return bundle.vconvert(cxc.sph3d)


result = process_bundle(bundle)  # Compiles successfully
```

**Best practice**: Keep bundle structure (field names and number of fields)
static for optimal JIT performance. The array values can vary, but the
dictionary keys should not change between compilations.

### Vectorization

Bundles support `vmap` naturally through indexing:

```
# Batch bundle
bundle_batch = cx.PointedVector(
    base=Vector.from_(jnp.array([[1, 2, 3], [4, 5, 6]]), "m"),
    velocity=Vector.from_(jnp.array([[10, 20, 30], [40, 50, 60]]), "m/s"),
)

# Index gives sub-vectors
sub0 = bundle_batch[0]
sub1 = bundle_batch[1]
```

## Integration with Frames

`PointedVector` stores vectors without reference frames. To attach frames, use
`coordinax.Coordinate`:

```
import coordinax as cx

# Create bundle
bundle = cx.PointedVector(base=..., velocity=...)

# Wrap in coordinate with frame
from coordinax.frames import ICRS

coord = cx.Coordinate(bundle, ICRS())

# Or create frame-aware vectors first, then bundle them
```

The bundle does not re-implement frame functionality - it complements the frame
system by managing base point dependencies.

## Best Practices

### 1. **Keep Structure Static for JIT**

```
# ✓ Good: Fixed field names
for data in dataset:
    bundle = cx.PointedVector(base=data.phys_disp, velocity=data.phys_vel)
    process_jit(bundle)

# ✗ Avoid: Changing field structure
for data in dataset:
    fields = {"velocity": data.phys_vel}
    if data.has_acc:
        fields["acceleration"] = data.phys_acc  # Structure changes
    bundle = cx.PointedVector(base=data.phys_disp, **fields)
```

### 2. **Use Appropriate Roles**

- `Point` → only for base
- `PhysDisp`, `PhysVel`, `PhysAcc` → for tangent vectors
- Future: `Momentum`, `Force`, etc. as needed

### 3. **Physical Components**

Always use physical components (uniform dimensions) for tangent vectors, never
coordinate differentials:

```
# ✓ Velocity in spherical: all m/s
v_sph = {
    "r": u.Q(1.0, "m/s"),
    "theta": u.Q(0.5, "m/s"),  # Physical, not d(theta)/dt
    "phi": u.Q(0.2, "m/s"),  # Physical, not d(phi)/dt
}

# ✗ Coordinate differentials
v_wrong = {
    "r": u.Q(1.0, "m/s"),
    "theta": u.Q(0.01, "rad/s"),  # Coordinate derivative
    "phi": u.Q(0.005, "rad/s"),
}
```

### 4. **Understand Automatic `at=` Handling**

The bundle automatically computes:

```
# For each field v:
at = bundle.base.vconvert(v.chart)  # Position conversion
v_new = v.vconvert(target_chart, at)
```

This ensures the base is in the correct representation before transformation.

## API Reference

See [API Documentation](../api/vecs.md) for detailed method signatures.

### Key Methods

- `__init__(*, base, **fields)` - Create bundle
- `from_(data, *, base=None)` - Create from mapping
- `vconvert(to_chart, *, field_charts=None)` - Convert bundle
- `__getitem__(key)` - Access fields or index batch
- `.base` / `.q` - Access base point
- `.keys()`, `.values()`, `.items()` - Mapping interface

## See Also

- [Vectors Guide](vectors.md) - Understanding roles and representations
- [Coordinate Representations](charts.md) - Available coordinate systems
- [Operators](operators.md) - Geometric operations on vectors
- API: `coordinax.PointedVector`
- API: `coordinax.transforms.physical_tangent_transform`
