# Vector Algebra: Points vs Displacements

This guide explains the distinction between **positions** (points) and
**displacements** (tangent vectors / free vectors) in `coordinax`, and how
vector addition follows the mathematical rules of affine geometry.

## Mathematical Background

In differential geometry and physics, we distinguish between:

- **Points (Positions)**: Elements of a manifold $M$. In Euclidean space, we
  often identify them with "position vectors from the origin," but this
  identification is coordinate-dependent.

- **Displacements (Tangent Vectors)**: Elements of a tangent space $T_p M$. In
  Euclidean space, displacements are "free vectors" that can be applied at any
  point.

The key insight is that **points cannot be added to each other**, but **a
displacement can be added to a point** to yield a new point.

### Critical Distinction: Transformation Rules

**This is the most important concept**: positions and displacements transform
differently under coordinate changes.

| Role           | Geometric Object              | Transformation Rule                            | Function                     | Base Point? |
| -------------- | ----------------------------- | ---------------------------------------------- | ---------------------------- | ----------- |
| `Point`        | Point $p \in M$               | Position transform: $p_S = f_{R \to S}(p_R)$   | `point_transform`            | No          |
| `PhysDisp`     | Physical vector $v \in T_p M$ | Tangent transform: $v_S = B_S(p)^T B_R(p) v_R$ | `physical_tangent_transform` | Sometimes\* |
| `PhysVel`      | Physical vector $v \in T_p M$ | Tangent transform: $v_S = B_S(p)^T B_R(p) v_R$ | `physical_tangent_transform` | Sometimes\* |
| `PhysAcc`      | Physical vector $a \in T_p M$ | Tangent transform: $a_S = B_S(p)^T B_R(p) v_R$ | `physical_tangent_transform` | Sometimes\* |

\*Base point required for:

- Embedded manifolds (e.g., sphere embedded in $\mathbb{R}^3$)
- Intrinsic manifolds (e.g., abstract Riemannian manifolds)
- NOT required for Euclidean spaces

### Conversion Policy Details

The distinction between these transformation types is formalized in coordinax
through two separate functions:

1. **`point_transform(to_chart, from_chart, p)`**: Chart-to-chart mapping
   - Maps points between coordinate charts: $p_{\text{new}} = f(p_{\text{old}})$
   - Used for: Position vectors (role `PhysDisp`)
   - Does not require a base point
   - Example: Converting a position from Cartesian to spherical coordinates

2. **`physical_tangent_transform(to_chart, from_chart, v, at=p_base)`**:
   Frame-based mapping at a point
   - Maps tangent vectors at a point via the frame transformation:
     $v_S = B_S(p)^T B_R(p) v_R$
   - Used for: Displacements, velocities, accelerations (roles `PhysDisp`,
     `PhysVel`, `PhysAcc`)
   - Requires base point `at=p_base` for non-Euclidean spaces
   - Example: Converting a velocity vector from cylindrical to Cartesian
     coordinates

**When is a base point required?**

- **Position transforms**: Never require a base point (they transform the point
  itself)
- **Tangent transforms**:
  - Required for embedded manifolds (vectors lie in tangent space)
  - Required for intrinsic manifolds (curvature affects vector transformations)
  - NOT required for Euclidean spaces (tangent space is isomorphic to the space
    itself)

**Key points:**

- **Positions** are points → transform via position map (coordinate
  substitution)
- **Displacements, velocities, accelerations** are physical vectors in
  orthonormal frames → transform via tangent map (frame transformation at a base
  point)
- Displacements have the **same transformation rule** as velocities, just
  different physical units (length vs length/time)

### Physical Components: Uniform Units

**CRITICAL**: PhysDisp, PhysVel, and PhysAcc store **physical vector components in
an orthonormal frame**, NOT coordinate increments. All components must have
uniform physical dimension.

In cylindrical coordinates (ρ, φ, z):

- **Physical Displacement**: `(rho=1m, phi=2m, z=3m)` where `phi` is the
  physical tangential length ✓
- **NOT coordinate increments**: `(Δrho=1m, Δphi=0.5rad, Δz=3m)` ✗

The `phi` component with value `2m` means "2 meters in the tangential direction
at the base point", NOT "2 radians of angular displacement".

### Example: Cylindrical → Cartesian

**Position transformation** (coordinate map):

$$
\begin{aligned}
x &= \rho \cos \phi \\
y &= \rho \sin \phi
\end{aligned}
$$

**Physical vector transformation** (frame transformation at point
$(\rho, \phi)$):

The orthonormal frame in cylindrical coordinates is:

$$
\begin{aligned}
\hat{e}_\rho &= (\cos\phi, \sin\phi, 0) \\
\hat{e}_\phi &= (-\sin\phi, \cos\phi, 0) \\
\hat{e}_z &= (0, 0, 1)
\end{aligned}
$$

A displacement with components $(v_\rho, v_\phi, v_z)$ transforms to Cartesian
as:

$$
\begin{aligned}
v_x &= v_\rho \cos\phi - v_\phi \sin\phi \\
v_y &= v_\rho \sin\phi + v_\phi \cos\phi \\
v_z &= v_z
\end{aligned}
$$

Note that the transformation depends on the base point $\phi$, unlike position
transformation.

## Vector Addition

### PhysDisp + PhysDisp → PhysDisp

Two displacements can be added to yield another displacement:

```
import coordinax as cx
import unxt as u

d1 = cx.Vector(
    {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
    cx.charts.cart3d,
    cx.charts.displacement,
)
d2 = cx.Vector(
    {"x": u.Q(0.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(0.0, "m")},
    cx.charts.cart3d,
    cx.charts.displacement,
)

result = d1.add(d2)
# result.role is PhysDisp
# result["x"] = 1.0 m, result["y"] = 2.0 m
```

### Position + Displacement → Position

A displacement can be applied to a position to get a new position (affine
translation):

```
import coordinax as cx
import unxt as u

pos = cx.Vector(
    {"x": u.Q(0.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
    cx.charts.cart3d,
    cx.roles.phys_disp,
)
disp = cx.Vector(
    {"x": u.Q(1.0, "m"), "y": u.Q(2.0, "m"), "z": u.Q(3.0, "m")},
    cx.charts.cart3d,
    cx.charts.displacement,
)

new_pos = pos.add(disp)
# new_pos.role is Pos
# new_pos represents the point (1, 2, 3) m
```

### Forbidden Operations

The following operations raise `TypeError` because they are mathematically
undefined:

```
import coordinax as cx
import unxt as u

pos1 = cx.Vector.from_([1, 0, 0], "m")
pos2 = cx.Vector.from_([0, 1, 0], "m")
disp = cx.Vector(
    {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")},
    cx.charts.cart3d,
    cx.charts.displacement,
)

# Position + Position: TypeError
# pos1.add(pos2)  # Raises: "Cannot add Position + Position"

# Displacement + Position: TypeError (use Position + Displacement instead)
# disp.add(pos1)  # Raises: "Cannot add Displacement + Position"
```

## Converting Positions to Displacements

Use `as_pos` to convert a position to a displacement relative to an origin:

```
import coordinax as cx
import unxt as u

pos = cx.Vector.from_([3, 4, 5], "m")

# From coordinate origin (automatically converts to Cartesian first)
disp = cx.as_pos(pos)
# disp represents the displacement (3, 4, 5) m from the origin
# Result is in Cartesian representation

# From an explicit origin
origin = cx.Vector.from_([1, 1, 1], "m")
disp = cx.as_pos(pos, origin)
# disp represents the displacement (2, 3, 4) m from origin
```

**Important:** When `origin=None` (Euclidean case), the position is first
converted to Cartesian using the position transform, then interpreted as a
displacement from the Cartesian origin. This ensures correct handling of
non-Cartesian coordinates:

```
import jax.numpy as jnp

# Position in spherical coordinates
pos_sph = cx.Vector(
    {"r": u.Q(2.0, "m"), "theta": u.Q(jnp.pi / 2, "rad"), "phi": u.Q(0.0, "rad")},
    cx.charts.sph3d,
    cx.roles.phys_disp,
)

# Displacement from origin (converted to Cartesian: x=2, y=0, z=0)
disp = cx.as_pos(pos_sph)
# disp.chart is cart3d
# disp["x"] = 2.0 m
```

**Requesting a specific representation:**

Use the `rep` parameter to convert the resulting displacement to a different
representation (uses `PhysDisp.vconvert` with tangent transform):

```
pos = cx.Vector.from_([1, 0, 0], "m")
origin = cx.Vector.from_([0, 0, 0], "m")

# Get displacement in spherical representation
disp_sph = cx.as_pos(pos, origin, chart=cx.charts.sph3d, at=pos)
# Uses physical_tangent_transform to convert to spherical at base point 'pos'
```

## Euclidean vs Non-Euclidean Representations

In **Euclidean spaces** (Cartesian, spherical, cylindrical coordinates on
$\mathbb{R}^n$), displacements are "free vectors" and can be added without
specifying a base point.

On **non-Euclidean manifolds** (e.g., a sphere), displacements live in tangent
spaces at specific points. Addition requires specifying the base point via the
`at=` parameter:

```
import coordinax as cx

# Check if a representation is Euclidean
cx.charts.cart3d.is_euclidean  # True
cx.charts.sph3d.is_euclidean  # True (spherical coords on R^3)
cx.charts.twosphere.is_euclidean  # False (intrinsic 2-sphere)
```

For non-Euclidean representations, use `Vector.add(other, at=base_point)`:

```
# On embedded manifolds (not yet fully implemented):
# result = d1.add(d2, at=base_point)
```

## Velocity and Acceleration

Velocity (`PhysVel`) and acceleration (`PhysAcc`) vectors represent time derivatives and
follow standard vector addition rules (same role + same role):

```
import coordinax as cx
import unxt as u

v1 = cx.Vector(
    {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
    cx.charts.cart3d,
    cx.roles.phys_vel,
)
v2 = cx.Vector(
    {"x": u.Q(0.0, "m/s"), "y": u.Q(2.0, "m/s"), "z": u.Q(0.0, "m/s")},
    cx.charts.cart3d,
    cx.roles.phys_vel,
)

v_total = v1.add(v2)
# v_total.role is Vel
```

## Summary

| Operation                     | Result         | Allowed?                          |
| ----------------------------- | -------------- | --------------------------------- |
| `PhysDisp + PhysDisp`         | `PhysDisp`     | ✅                                |
| `Point + PhysDisp`            | `PhysDisp`     | ✅                                |
| `PhysDisp + Point`            | —              | ❌ Use `Point + PhysDisp`         |
| `Point + Point`               | —              | ❌ Subtract to get `PhysDisp`     |
| `PhysVel + PhysVel`           | `PhysVel`      | ✅                                |
| `PhysAcc + PhysAcc`           | `PhysAcc`      | ✅                                |

---

:::{seealso}

- [Vectors Guide](vectors.md) - Creating and converting vectors
- [Representations Guide](charts.md) - Coordinate systems
- [Metrics Guide](metrics.md) - Metric tensors and distances

:::
