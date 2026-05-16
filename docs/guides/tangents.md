# Working With Tangent Vectors

This guide explains `Tangent` — coordinax's type for **tangent-space quantities** (velocities, displacements, accelerations). For bundling a `Tangent` with a base `Point`, see the [Coordinate guide](./vectors.md) and the [Coordinate tutorial](../tutorials/coordinate_objects.md). For API reference see [the vector module reference](../api/vectors.md).

## Why Tangent Vectors?

Points and tangent vectors look similar — both are dictionaries of components attached to a chart — but they **transform differently** when you change coordinate system:

- **Point** `(r, θ, φ)`: transforms by the chart transition map $\varphi' \circ \varphi^{-1}$.
- **Tangent** `(ṙ, θ̇, φ̇)`: transforms by the **Jacobian** $J^j{}_i = \partial \tilde{q}^j / \partial q^i$ evaluated at the base point.

Getting this wrong produces incorrect physical results. `Tangent` encodes the correct transformation law, so `cconvert` always does the right thing automatically.

Additionally, tangent vectors carry two extra pieces of metadata:

- **basis**: are the components expressed in the _coordinate basis_ ($\partial/\partial q^i$) or the _physical (orthonormal) basis_ ($\hat{e}_i$)?
- **semantic**: what physical quantity do the components represent — `vel` (velocity), `dpl` (displacement), or `acc` (acceleration)?

## Basis And Semantic Kind

### Basis

| Object | Singleton | Meaning |
| --- | --- | --- |
| `CoordinateBasis` | `cxr.coord_basis` | components in coordinate (tangent) basis |
| `PhysicalBasis` | `cxr.phys_basis` | components in orthonormal physical frame |

For **Cartesian charts** these are identical. For **spherical/cylindrical** charts the coordinate basis has components with units like `m/s`, `rad/s`, `rad/s`, while the physical basis has all components in `m/s` (scaled by the metric's scale factors).

### Semantic Kind

| Object         | Singleton | Physical meaning                  |
| -------------- | --------- | --------------------------------- |
| `Displacement` | `cxr.dpl` | position displacement $\Delta q$  |
| `Velocity`     | `cxr.vel` | time derivative $\dot{q}$         |
| `Acceleration` | `cxr.acc` | second time derivative $\ddot{q}$ |

### Pre-Built Representations

The most common combinations are available as ready-made singletons:

| Name             | `geom_kind`       | `basis`       | `semantic` |
| ---------------- | ----------------- | ------------- | ---------- |
| `cxr.coord_disp` | `TangentGeometry` | `coord_basis` | `dpl`      |
| `cxr.coord_vel`  | `TangentGeometry` | `coord_basis` | `vel`      |
| `cxr.coord_acc`  | `TangentGeometry` | `coord_basis` | `acc`      |
| `cxr.phys_disp`  | `TangentGeometry` | `phys_basis`  | `dpl`      |
| `cxr.phys_vel`   | `TangentGeometry` | `phys_basis`  | `vel`      |
| `cxr.phys_acc`   | `TangentGeometry` | `phys_basis`  | `acc`      |

## Constructing Tangent Vectors

::::{tab-set}

:::{tab-item} Dict + rep (most explicit)

Name every component and supply the representation directly:

```{code-block} python
>>> import coordinax.main as cx
>>> import coordinax.charts as cxc
>>> import coordinax.representations as cxr
>>> import unxt as u

>>> vel = cx.Tangent.from_(
...    {"x": u.Q(1, "m/s"), "y": u.Q(2, "m/s"), "z": u.Q(3, "m/s")},
...    cxc.cart3d, cxr.coord_vel)
>>> print(vel)
<Tangent: chart=Cart3D (x, y, z) [m / s]
    [1 2 3]>
```

:::

:::{tab-item} Dict + basis + semantic

Specify basis and semantic separately (equivalent to the above):

```{code-block} python
>>> vel = cx.Tangent.from_(
...    {"x": u.Q(1, "m/s"), "y": u.Q(2, "m/s"), "z": u.Q(3, "m/s")},
...    cxc.cart3d, cxr.coord_basis, cxr.vel)
>>> print(vel)
<Tangent: chart=Cart3D (x, y, z) [m / s]
    [1 2 3]>
```

:::

:::{tab-item} Array + unit

Chart is inferred from the array length (3 → `cart3d`):

```{code-block} python
>>> vel = cx.Tangent.from_([1, 2, 3], "m/s")
>>> print(vel.chart)  # Cart3D(M=Rn(3))
Cart3D[('x', 'y', 'z'), ('length', 'length', 'length')](M=Rn(3))
```

:::

:::{tab-item} Passthrough

Passing an existing `Tangent` returns it unchanged:

```{code-block} python
>>> vel2 = cx.Tangent.from_(vel)
>>> assert vel2 is vel
```

:::

::::

## Accessing Data

```{code-block} python
>>> vel = cx.Tangent.from_(
...    {"x": u.Q(1, "m/s"), "y": u.Q(2, "m/s"), "z": u.Q(3, "m/s")},
...    cxc.cart3d, cxr.coord_vel)

>>> # Component access by name
>>> print(vel["x"])  # Q(1., 'm / s')
Q['speed'](Array(1, dtype=int64, weak_type=True), unit='m / s')

# Metadata
>>> print(vel.chart)  # Cart3D(M=Rn(3))
Cart3D[('x', 'y', 'z'), ('length', 'length', 'length')](M=Rn(3))

>>> print(vel.basis)  # coord_basis
CoordinateBasis()

>>> print(vel.semantic)  # vel
Velocity()

>>> print(vel.frame)  # NoFrame()
NoFrame()
```

## Chart Conversion (Jacobian Pushforward)

Converting a `Tangent` to a new chart requires knowing the **base point** at which the Jacobian is evaluated. Pass it via the `at=` argument:

```{code-block} python
>>> point = cx.Point.from_([1, 0, 0], "m") # Base point in Cartesian
>>> vel_cart = cx.Tangent.from_(
...     {"x": u.Q(1, "m/s"), "y": u.Q(0, "m/s"), "z": u.Q(0, "m/s")},
...     cxc.cart3d, cxr.coord_vel)

>>> # Convert to spherical — Jacobian is evaluated at `point`
>>> vel_sph = vel_cart.cconvert(cxc.sph3d, at=point)
>>> print(vel_sph)
<Tangent: chart=Spherical3D (r[m / s], theta[rad / s], phi[rad / s])
    [ 1. -0.  0.]>
```

**Key point**: the `at=` base point must be in the **same chart** as the tangent vector. To convert a point and its tangent field together in one call, bundle them into a `Coordinate` — see the [Coordinate tutorial](../tutorials/coordinate_objects.md).

## Changing Basis

`change_basis` converts between coordinate and physical bases for `Tangent` objects directly. Pass the tangent vector, the target basis, and the base `Point` at which scale factors are evaluated:

```{code-block} python
>>> # Velocity in coordinate basis at a spherical point
>>> point_sph = cx.Point.from_(
...     {"r": u.Q(1, "m"), "theta": u.Q(0.5, "rad"), "phi": u.Q(0, "rad")}, cxc.sph3d
... )
>>> vel_sph_coord = cx.Tangent.from_(
...     {"r": u.Q(1, "m/s"), "theta": u.Q(0, "rad/s"), "phi": u.Q(0, "rad/s")},
...     cxc.sph3d, cxr.coord_vel)
>>> # Convert to physical basis (all components in m/s)
>>> vel_sph_phys = cxr.change_basis(vel_sph_coord, cxr.phys_basis, at=point_sph)
>>> print(vel_sph_phys.rep)  # phys_vel
Representation(
    geom_kind=TangentGeometry(), basis=PhysicalBasis(), semantic_kind=Velocity()
)
```

## Promoting a Point to a Displacement

`change_basis` has a second role: it can **promote a `Point` to a `Tangent` with `Displacement` semantics**. This is useful when you have a position vector that you want to reinterpret as a tangent-space displacement — for example, when computing offsets or feeding a position into an operation that expects a displacement.

The component data is unchanged; only the geometric interpretation is recast from a manifold point (`PointGeometry`) to a tangent-space displacement (`TangentGeometry`, `Displacement`).

```{code-block} python
>>> pt = cx.Point.from_([1, 2, 3], "m")

>>> # Promote to a coordinate-basis displacement
>>> disp = cxr.change_basis(pt, cxr.coord_basis)
>>> print(disp)
<Tangent: chart=Cart3D (x, y, z) [m]
    [1 2 3]>
```

You can also request the physical basis directly:

```{code-block} python
>>> disp_phys = cxr.change_basis(pt, cxr.phys_basis)
>>> print(disp_phys.rep)   # phys_disp
Representation(
    geom_kind=TangentGeometry(), basis=PhysicalBasis(), semantic_kind=Displacement()
)
```

The `at` and `usys` keyword arguments are accepted for API consistency but have no effect (no numerical transformation is performed).

### JIT

```{code-block} python
>>> import jax
>>> point = cx.Point.from_([1.0, 0.0, 0.0], "m")
>>> vel = cx.Tangent.from_(
...     {"x": u.Q(1.0, "m/s"), "y": u.Q(0.0, "m/s"), "z": u.Q(0.0, "m/s")},
...     cxc.cart3d, cxr.coord_vel)

>>> convert_vel = jax.jit(lambda v, p: cx.cconvert(v, cxc.sph3d, at=p))
>>> vel_sph = convert_vel(vel, point)
>>> print(vel_sph.chart)  # Spherical3D(M=Rn(3))
Spherical3D[('r', 'theta', 'phi'), ('length', 'angle', 'angle')](M=Rn(3))
```

### vmap

Use scalar-first design and batch with `vmap`:

```{code-block} python
>>> import jax.numpy as jnp

>>> # Build a batched Tangent by stacking scalar values per component
>>> batch_vel = cx.Tangent(
...     { "x": u.Q(jnp.array([1, 2, 3]), "m/s"),
...       "y": u.Q(jnp.zeros(3), "m/s"), "z": u.Q(jnp.zeros(3), "m/s") },
...     cxc.cart3d, cxr.coord_basis, cxr.vel)
>>> batch_point = cx.Point(
...     { "x": u.Q(jnp.ones(3), "m"), "y": u.Q(jnp.zeros(3), "m"),
...       "z": u.Q(jnp.zeros(3), "m") }, cxc.cart3d)

>>> vec_fn = jax.vmap(lambda v, p: cx.cconvert(v, cxc.sph3d, at=p))
>>> vels_sph = vec_fn(batch_vel, batch_point)
>>> print(vels_sph.data)  # Spherical3D(M=Rn(3))
{'phi': Q([0., 0., 0.], 'rad / s'), 'r': Q([1., 2., 3.], 'm / s'), 'theta': Q([-0., -0., -0.], 'rad / s')}
```

## When To Use Tangent

| You have | Use |
| --- | --- |
| Position only | `Point` — see [Point tutorial](../tutorials/point_objects.md) |
| Velocity / displacement / acceleration only | `Tangent` (this guide) |
| Position + tangent field(s) | `Coordinate` — see [Coordinate tutorial](../tutorials/coordinate_objects.md) |
| Position to reinterpret as a displacement | `change_basis(pt, cxr.coord_basis)` → `Tangent[Displacement]` |
