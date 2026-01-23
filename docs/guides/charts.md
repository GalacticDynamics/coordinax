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
- Role: semantic interpretation of vector data (Pos, Vel, PhysAcc, etc.). A role is
  not a rep.
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
import coordinax as cx
import unxt as u

d = cx.distances.Distance(10.0, "kpc")
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
import coordinax as cx
import unxt as u

rep_cart = cx.charts.cart3d
rep_sph = cx.charts.sph3d

q = {
    "x": u.Q(1.0, "km"),
    "y": u.Q(2.0, "km"),
    "z": u.Q(3.0, "km"),
}

q_sph = cx.transforms.point_transform(rep_sph, rep_cart, q)
# Result: {'r': Quantity(..., unit='km'), 'theta': ..., 'phi': ...}
```

### With Bare Arrays

When the `CsDict` contains bare arrays (JAX arrays, NumPy arrays, or Python
scalars), the function operates on dimensionless values:

```
import coordinax as cx
import unxt as u

# 1D example: Radial to Cartesian
to_chart, from_chart = cx.charts.cart1d, cx.charts.radial1d
cx.transforms.point_transform(to_chart, from_chart, {"r": 5})
# Result: {'x': 5}

# 2D example: Polar to Cartesian
import jax.numpy as jnp
p_polar = {"r": 1.0, "theta": jnp.pi / 4}
p_cart = cx.transforms.point_transform(cx.charts.cart2d, cx.charts.polar2d, p_polar,
                                       usys=u.unitsystems.galactic)
# Result: {'x': 0.707..., 'y': 0.707...}
```

This is useful for quick calculations where units are not needed, or when
working with normalized/dimensionless coordinates.

## Embeddings and Metrics

```
import coordinax as cx
import unxt as u

embed = cx.charts.EmbeddedManifold(
    chart_kind=cx.charts.twosphere,
    ambient_kind=cx.charts.cart3d,
    params={"R": u.Q(1.0, "km")},
)

p = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.5, "rad")}
q_emb = cx.embeddings.embed_point(embed, p)
p_back = cx.embeddings.project_point(embed, q_emb)

metric = cx.metrics.metric_of(cx.charts.twosphere)
g = metric.metric_matrix(cx.charts.twosphere, p)
```

Metric defaults:

- Euclidean charts exposed in `cx.charts.cart3d` use the Euclidean metric by
  default.
- `TwoSphere` uses the intrinsic sphere metric.
- `SpaceTimeCT` uses the Minkowski metric.
- `SpaceTimeEuclidean` uses a Euclidean metric in 4D.

## Vectors and Roles

```
import coordinax as cx
import unxt as u

q = {
    "x": u.Q(1.0, "km"),
    "y": u.Q(2.0, "km"),
    "z": u.Q(3.0, "km"),
}
v = {
    "x": u.Q(4.0, "km/s"),
    "y": u.Q(5.0, "km/s"),
    "z": u.Q(6.0, "km/s"),
}

qvec = cx.Vector(data=q, chart=cx.charts.cart3d, role=cx.roles.PhysDisp())
vvec = cx.Vector(data=v, chart=cx.charts.cart3d, role=cx.roles.PhysVel())

q_sph = qvec.vconvert(cx.charts.sph3d)
v_sph = vvec.vconvert(cx.charts.sph3d, qvec)
```
