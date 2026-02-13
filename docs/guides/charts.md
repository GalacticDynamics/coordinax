---
title: Representations and Metrics
---

# Representations and Metrics

This guide introduces representations (coordinate charts), metrics, and roles,
and shows how to transform parameter dictionaries and vectors.

## Terminology

- Representation (Rep): a coordinate chart / component schema (names + physical
  dimensions). A rep does not store numerical values.
- Vector: data + rep + role. Data are the coordinate values or physical
  components.
- Role: semantic interpretation of vector data (Pos, Vel, PhysAcc, etc.). A role
  is not a rep.
- Metric: a bilinear form g on the tangent space defining inner products and
  norms (Euclidean, sphere intrinsic, Minkowski).
- Physical components: components of a geometric vector expressed in an
  orthonormal frame with respect to the active metric. Components have uniform
  units (e.g. all speed or all acceleration).
- Coordinate derivatives: time derivatives of coordinate components (e.g.
  \dot\theta, \ddot\phi); these may have heterogeneous units and are not what
  `physical_tangent_transform`/`vconvert` uses for physical vectors.

## Distances and Angles

```
import coordinax.distances as cxd
import unxt as u

d = cxd.Distance(10.0, "kpc")
a = u.Angle(30.0, "deg")
```

## Representations and Coordinate Maps

The `point_transform` function converts coordinates between different charts. It
accepts `CsDict` values that can be either **Quantities** (values with explicit
units) or **bare arrays** (dimensionless values).

### With Quantities

When the `CsDict` contains `unxt.Quantity` objects, units are handled
automatically:

```
import coordinax.charts as cxc
import coordinax.transforms as cxt

rep_cart = cxc.cart3d
rep_sph = cxc.sph3d

q = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}

q_sph = cxt.point_transform(rep_sph, rep_cart, q)
# Result: {'r': Quantity(..., unit='km'), 'theta': ..., 'phi': ...}
```

### With Bare Arrays

When the `CsDict` contains bare arrays (JAX arrays, NumPy arrays, or Python
scalars), the function operates on dimensionless values:

```
import coordinax as cx
import unxt as u

# 1D example: Radial to Cartesian
to_chart, from_chart = cxc.cart1d, cxc.radial1d
cxt.point_transform(to_chart, from_chart, {"r": 5})
# Result: {'x': 5}

# 2D example: Polar to Cartesian
import jax.numpy as jnp
p_polar = {"r": 1.0, "theta": jnp.pi / 4}
p_cart = cxt.point_transform(cxc.cart2d, cxc.polar2d, p_polar, usys=u.unitsystems.galactic)
# Result: {'x': 0.707..., 'y': 0.707...}
```

This is useful for quick calculations where units are not needed, or when
working with normalized/dimensionless coordinates.

## Embeddings and Metrics

```
import coordinax.embeddings as cxe
import coordinax.metrics as cxm
import unxt as u

embed = cxe.EmbeddedManifold(chart_kind=cxc.twosphere, ambient_kind=cxc.cart3d,
                             params={"R": u.Q(1.0, "km")})

p = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.5, "rad")}
q_emb = cxe.embed_point(embed, p)
p_back = cxe.project_point(embed, q_emb)

metric = cxm.metric_of(cxc.twosphere)
g = metric.metric_matrix(cxc.twosphere, p)
```

Metric defaults:

- Euclidean charts exposed in `cxc.cart3d` use the Euclidean metric by default.
- `TwoSphere` uses the intrinsic sphere metric.
- `SpaceTimeCT` uses the Minkowski metric.
- `SpaceTimeEuclidean` uses a Euclidean metric in 4D.

## Vectors and Roles

```
import coordinax as cx
import unxt as u

q = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
v = {"x": u.Q(4, "km/s"), "y": u.Q(5, "km/s"), "z": u.Q(6, "km/s"),}

qvec = cx.Vector(data=q, chart=cxc.cart3d, role=cxr.PhysDisp())
vvec = cx.Vector(data=v, chart=cxc.cart3d, role=cxr.PhysVel())

q_sph = qvec.vconvert(cxc.sph3d)
v_sph = vvec.vconvert(cxc.sph3d, qvec)
```
