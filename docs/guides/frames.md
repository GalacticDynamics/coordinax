# Working With Frames

This guide provides a conceptual introduction to coordinax reference frames and transformation groups, with practical workflows. For API reference, see [the frames module reference](../api/frames.md).

## Motivation: Why Reference Frames?

In a single inertial reference frame, coordinates are simple: a point is a point, and vectors obey standard rules.

Real problems often involve **multiple observers**:

- **N-body simulations**: sun-centered, galactocentric, and particle-relative frames coexist
- **Galactic dynamics**: ICRS (celestial), Galactocentric (rotating), and stream frames
- **Spacecraft**: inertial frame, spacecraft body frame, instrument frame
- **Relativistic physics**: different time-orthogonal hypersurfaces

Rather than manually converting every coordinate every time, coordinax allows you to **define frames once** and **relate them via transformations**.

## The Reference Frame Model

A **reference frame** $F$ represents a spatial observer's perspective on coordinates.

**Active (Moving) Frame View**:

$$
\text{Point } p \xrightarrow{\text{Observe in frame } F} \text{Coordinates } q = \varphi_F(p)
$$

Different frames produce different coordinate representations of the same point:

$$
q_F = \varphi_F(p), \quad q_{F'} = \varphi_{F'}(p)
$$

**Frame Transition (Active Transformation)**:

A transformation $T: F \to F'$ acts on coordinates by the operator attached to that frame change:

$$
q_{F'} = T(q_F)
$$

In `coordinax`, this is an **active** transformation: applying the operator moves the represented point data on the manifold.

## Transformation Groups: Mathematical Classification

Transformations are classified by the **geometric structures they preserve**. This classification lives in **transformation groups**.

### Group Hierarchy (ASCII Tree)

```text
DiffeomorphismGroup
├─ AffineGroup
│  ├─ EuclideanGroup
│  │  ├─ OrthogonalGroup
│  │  │  └─ SpecialOrthogonalGroup
│  │  └─ LorentzGroup
│  │     └─ ProperOrthochronousLorentzGroup
│  └─ PoincareGroup
└─ IdentityGroup
```

**Reading this tree**: arrows point toward _more restrictive_ groups. Moving down = stronger constraints.

### Group Semantics

#### **Identity Group**

- Preserves: everything
- Allows: only the identity map
- Math: $T(x) = x$
- Use: null placeholder

#### **Diffeomorphism Group**

- Preserves: smooth structure
- Allows: any smooth invertible map
- Math: $T: M \to M$, $T$ and $T^{-1}$ smooth
- Use: most general transformations

#### **Affine Group**

- Preserves: parallelism, ratios along lines
- Allows: linear map + translation
- Math: $T(x) = A x + b$ with $A$ invertible
- Use: coordinate systems with linear structure
- Examples: shearing, scaling, rotation, translation

#### **Euclidean Group** (Rigid Motions)

- Preserves: **distances** $\|T(x) - T(y)\| = \|x - y\|$, **angles**
- Allows: rotations, translations, reflections
- Math: Orthogonal $Q$ (preserves metric) + translation $b$: $T(x) = Qx + b$
- Use: rigid body motion, non-expanding cosmologies
- Examples: spacecraft body-frame rotations, galactic proper motion

#### **Orthogonal Group**

- Preserves: angles, inner product $\langle T(x), T(y) \rangle = \langle x, y \rangle$
- Allows: rotations and reflections about origin
- Math: $Q^T Q = I$ (preserves metric)
- Use: rotations and reflections with fixed origin (no translation)
- Examples: coordinate system alignment

#### **Special Orthogonal Group** (SO(n))

- Preserves: distances, angles, **orientation** (handedness)
- Allows: **proper rotations only** (no reflections)
- Math: $Q^T Q = I$ and $\det(Q) = +1$
- Use: when orientation matters (cross products, angular momentum)
- Examples: spinning reference frames, gyroscope orientation

#### **Lorentz Group** (Special Relativity)

- Preserves: **spacetime intervals** $|ds|^2 = -(dt)^2 + (dx)^2 + (dy)^2 + (dz)^2$
- Allows: boosts (Lorentz transformations) mixing space and time
- Math: $\Lambda^T \eta \Lambda = \eta$ (preserves Minkowski metric)
- Use: relativistic coordinate transformations
- Examples: reference frames moving at relativistic speeds

#### **Proper Orthochronous Lorentz Group**

- Preserves: spacetime intervals, **spatial and temporal orientations**
- Allows: only "physical" Lorentz transformations
- Math: Lorentz transforms with $\det(\Lambda) = +1$ and forward time-direction
- Use: real-world relativistic transformations
- Examples: actual spacecraft boosts, particle collision frames

#### **Poincaré Group**

- Preserves: spacetime intervals (like Lorentz + translations)
- Allows: Lorentz transformations + spacetime translations
- Math: Semidirect product of Lorentz group and spacetime translation
- Use: most general relativistic frame transitions
- Examples: combining boosts and general spacetime repositioning

### Why Groups Matter

Group membership answers: **"What geometric properties does this transformation preserve?"**

```python
# Example: Are distances preserved?
# Yes if: transform ∈ Euclidean ⊂ Affine
# No if: transform ∈ Affine ⊂ Diffeomorphism
```

This enables:

1. **Correctness**: ensure physically meaningful transforms
2. **Dispatch**: select correct numerical methods based on group
3. **Optimization**: simplify or cancel transforms knowing properties

## Building Transformations

### Primitive Transforms

#### Identity (Do Nothing)

```python
import coordinax.frames as cxf
import coordinax.transforms as cxfm

t_id = cxfm.Identity()
# cxf.identity is the same instance: Identity()
```

#### Translation (Displacement)

```python
import coordinax.frames as cxf
import coordinax.charts as cxc
import coordinax.transforms as cxfm
import unxt as u

# Translate by (1, 0, 0)
t_translate = cxfm.Translate({"x": 1, "y": 0, "z": 0}, chart=cxc.cart3d)
```

This is in the **Euclidean group** (preserves distances and angles).

#### Rotation

```python
import jax.numpy as jnp
import math

# Rotate around z-axis by π/2
theta = math.pi / 2
R = jnp.array(
    [
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
t_rotate = cxfm.Rotate(R)
```

This is in the **Special Orthogonal group** (proper rotations, orientation-preserving).

#### Reflection

```python
import coordinax.transforms as cxfm
import unxt as u

# Reflect across the yz-plane
t_reflect = cxfm.Reflect.from_normal([1.0, 0.0, 0.0])

q = u.Q([1.0, 2.0, 3.0], "km")
cxfm.act(t_reflect, None, q)
# Q([-1.,  2.,  3.], 'km')
```

This is in the **Orthogonal group** (distance-preserving, orientation-reversing).

### Composition (Chaining Transforms)

Use the `|` operator to compose:

```python
import coordinax.charts as cxc
import coordinax.transforms as cxfm
import jax.numpy as jnp
import math

t1 = cxfm.Translate({"x": 1, "y": 0, "z": 0}, chart=cxc.cart3d)
theta = math.pi / 2
R = jnp.array(
    [
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
t2 = cxfm.Rotate(R)

# Compose: apply t1 first, then t2
composed = t2 | t1
```

**Evaluation order** (right-to-left):

$$
\text{result} = T_2(T_1(x))
$$

### Inversion (Reversing)

```python
# Inverse transforms are not yet available via cxf.inverse().
# Construct the reverse manually:
t_original = cxfm.Translate({"x": 1, "y": 0, "z": 0}, chart=cxc.cart3d)
t_inverse = cxfm.Translate({"x": -1, "y": 0, "z": 0}, chart=cxc.cart3d)

# These cancel out:
cancelled = t_inverse | t_original  # Equivalent to identity
```

### Simplification (Optimization)

```python
import coordinax.frames as cxf
import coordinax.charts as cxc
import coordinax.transforms as cxfm
import jax.numpy as jnp
import math

# Build a complex composition
t1 = cxfm.Translate({"x": 1, "y": 0, "z": 0}, chart=cxc.cart3d)
theta = math.pi / 2
R = jnp.array(
    [
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
t2 = cxfm.Rotate(R)
t3 = cxfm.Translate({"x": -1, "y": 0, "z": 0}, chart=cxc.cart3d)

complex_transform = t3 | t2 | t1

# Simplify: reduces nesting and cancels identities
simplified = cxfm.simplify(complex_transform)

# Both are mathematically equivalent, but simplified is more efficient
```

## Working With Reference Frames

A reference frame defines a spatial observer. Built-in example frames:

```python
import coordinax.frames as cxf

alice = cxf.Alice()  # stationary frame at origin
alex = cxf.Alex()  # stationary frame offset from Alice
no_frame = cxf.NoFrame()  # identity (null) frame
```

### Frame Transitions

Relate two frames via a transformation:

```python
# Get the transformation FROM alice TO alex
transform_alice_to_alex = cxf.frame_transition(alice, alex)

# This is a transform operator that can be applied
import coordinax.main as cx

v_in_alice = cx.Point.from_([1, 2, 3], "m", cxc.cart3d)
v_in_alex = cxfm.act(transform_alice_to_alex, None, v_in_alice)
```

### Custom Frames

Define domain-specific reference frames by subclassing:

```python
# Custom frame sketch (illustrative — fill in frame_transition logic):
#
# import coordinax.frames as cxf
# from coordinax.frames import AbstractReferenceFrame, AbstractTransform
#
# class RotatingFrame(AbstractReferenceFrame):
#     """A frame rotating at constant angular velocity."""
#
#     omega: float = 1.0  # rad/s
#
#     def frame_transition(self,
#                          to_frame: AbstractReferenceFrame) -> AbstractTransform:
#         """Compute transform to another frame."""
#         ...
```

For astronomical applications, `coordinax.astro` provides pre-built frames:

```python
# (If coordinax.astro is installed)
import coordinax.astro as cxastro

icrs = cxastro.ICRS()  # Celestial reference frame
galactocentric = cxastro.Galactocentric()

# Transition between them
xform = cxf.frame_transition(icrs, galactocentric)
```

## Coordinate Objects In Frame Workflows

A `Coordinate` is a `Vector` attached to a reference frame. This is often the most direct way to express "this data is measured by this observer":

```python
import coordinax.main as cx
import coordinax.frames as cxf

coord = cx.Point.from_([1, 2, 3], "kpc", cxf.alice)
print(coord.frame)  # Alice()
print(coord.chart)  # Cart3D(M=Rn(3))
```

### Frame Transformations On Coordinates

Use `to_frame()` when you want to apply the operator associated with changing observers. In the active convention, this moves the represented point data into the target frame.

```python
import coordinax.main as cx
import coordinax.frames as cxf

coord_alice = cx.Point.from_([1, 2, 3], "m", cxf.alice)
coord_alex = coord_alice.to_frame(cxf.alex)

print(coord_alice.frame)  # Alice()
print(coord_alex.frame)  # Alex()
```

You can also compute and apply the frame transition operator explicitly:

```python
op = cxf.frame_transition(cxf.alice, cxf.alex)
coord_alex_2 = op(coord_alice)
```

### Frame Change vs Chart Change

`to_frame()` changes reference frame. `cconvert()` changes chart. These answer different questions and can be chained.

```python
import coordinax.charts as cxc

coord_cart = cx.Point.from_([1, 2, 3], "m", cxf.alice)
coord_sph = coord_cart.cconvert(cxc.sph3d)

print(coord_cart.frame, coord_cart.chart)  # Alice(), Cart3D(M=Rn(3))
print(coord_sph.frame, coord_sph.chart)  # Alice(), Spherical3D()
```

For a full frame-oriented workflow (constructor patterns, frame+chart pipelines, and JAX batching), see [Working With Vectors](./vectors.md#coordinate-objects).

## Practical Workflow: Rotating Frame

Here's a complete example: **observe a rotating planet from an inertial frame**.

```python
import coordinax.frames as cxf
import coordinax.main as cx
import coordinax.charts as cxc
import coordinax.transforms as cxfm
import jax.numpy as jnp
import math

# Define frames
inertial = cxf.alice  # Fixed reference frame

# Define rotation: planet rotates 0.1 rad around z-axis
# Build a rotation matrix explicitly
theta = 0.1  # radians
R = jnp.array(
    [
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
rotating_frame = cxf.TransformedReferenceFrame(inertial, cxfm.Rotate(R))

# Observe a point in the inertial frame
position_inertial = cx.Point.from_([1, 0, 0], "m", cxc.cart3d)

# Get the transition operator
xform = cxf.frame_transition(inertial, rotating_frame)

# Apply to get position in rotating frame (act takes op, tau, x)
position_rotating = cxfm.act(xform, None, position_inertial)

print(position_rotating.data)  # Different coordinates, same point
```

## JAX Integration Patterns

```python
# JAX integration sketch (illustrative):
#
# import jax
# import coordinax.frames as cxf
# import coordinax.main as cx
#
# @jax.jit
# def batch_transform_points(frame1, frame2, vectors):
#     transform = cxf.frame_transition(frame1, frame2)
#     return jax.vmap(lambda v: cxfm.act(transform, None, v))(vectors)
#
# Result is JIT-compiled and efficient
```

## Common Pitfalls

### 1. Composition Order

```python
# RIGHT: apply t1 first, then t2
result = t2 | t1

# WRONG: this applies t2 first
result = t1 | t2  # Different!
```

### 2. Active vs Passive

Coordinax uses **active** transformations:

```python
# Active: apply the frame-transition operator to the point data
# v_rotated = cxfm.act(rotate_transform, None, v)  # RIGHT (act takes op, tau, x)

# Passive language is only for comparison with other conventions
# In coordinax, think: "apply the transform" rather than "just relabel coordinates"
```

### 3. Forgetting to Simplify

```python
import coordinax.frames as cxf
import coordinax.charts as cxc
import coordinax.transforms as cxfm
import jax.numpy as jnp
import math

# Redefine for this example
_theta = math.pi / 2
_R = jnp.array(
    [
        [math.cos(_theta), -math.sin(_theta), 0.0],
        [math.sin(_theta), math.cos(_theta), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
_t1 = cxfm.Translate({"x": 1, "y": 0, "z": 0}, chart=cxc.cart3d)
_t2 = cxfm.Rotate(_R)
_t3 = cxfm.Translate({"x": -1, "y": 0, "z": 0}, chart=cxc.cart3d)
_t4 = cxfm.Translate({"x": 0, "y": 1, "z": 0}, chart=cxc.cart3d)
_t5 = cxfm.Translate({"x": 0, "y": -1, "z": 0}, chart=cxc.cart3d)
_t6 = cxfm.Translate({"x": 0, "y": 0, "z": 1}, chart=cxc.cart3d)

# Complex nested transforms are slow
complex_t = _t6 | _t5 | _t4 | _t3 | _t2 | _t1

# Simplify for performance
simple_t = cxfm.simplify(complex_t)
```

### 4. Group Constraints

Not all transforms are valid on all manifolds. Let the type system help:

```python
# A manifold might require Euclidean transforms only
# Attempting affine (shearing) might raise TypeError
```

## Summary: Workflow Pattern

1. **Define** frames and their relationships
2. **Compute** transformations via `frame_transition()`
3. **Apply** to vectors via `act()`
4. **Compose** multiple transformations with `|`
5. **Simplify** to optimize
6. **Vectorize** with `vmap` for batching
7. **Differentiate** with `grad` if needed

This workflow enables clean, composable code for multi-frame simulations.
