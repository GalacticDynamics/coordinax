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
- Role: semantic interpretation of vector data (Pos, Vel, Acc, etc.). A role is
  not a rep.
- Metric: a bilinear form g on the tangent space defining inner products and
  norms (Euclidean, sphere intrinsic, Minkowski).
- Physical components: components of a geometric vector expressed in an
  orthonormal frame with respect to the active metric. Components have uniform
  units (e.g. all speed or all acceleration).
- Coordinate derivatives: time derivatives of coordinate components (e.g.
  \dot\theta, \ddot\phi); these may have heterogeneous units and are not what
  diff_map/vconvert uses for physical vectors.

## Distances and Angles

```python
import coordinax as cx
import unxt as u

d = cx.distance.Distance(10.0, "kpc")
a = u.Angle(30.0, "deg")
```

## Representations and Coordinate Maps

```python
import coordinax as cx
import unxt as u

rep_cart = cx.r.cart3d
rep_sph = cx.r.sph3d

q = {
    "x": u.Quantity(1.0, "km"),
    "y": u.Quantity(2.0, "km"),
    "z": u.Quantity(3.0, "km"),
}

q_sph = cx.r.coord_map(rep_sph, rep_cart, q)
```

## Embeddings and Metrics

```python
import coordinax as cx
import unxt as u

embed = cx.r.EmbeddedManifold(
    chart_kind=cx.r.twosphere,
    ambient_kind=cx.r.cart3d,
    params={"R": u.Quantity(1.0, "km")},
)

p = {"theta": u.Angle(1.0, "rad"), "phi": u.Angle(0.5, "rad")}
q_emb = cx.r.embed_pos(embed, p)
p_back = cx.r.project_pos(embed, q_emb)

metric = cx.r.metric_of(cx.r.twosphere)
g = metric.metric_matrix(cx.r.twosphere, p)
```

Metric defaults:

- Euclidean charts exposed in `cx.r` use the Euclidean metric by default.
- `TwoSphere` uses the intrinsic sphere metric.
- `SpaceTimeCT` uses the Minkowski metric.
- `SpaceTimeEuclidean` uses a Euclidean metric in 4D.

## Vectors and Roles

```python
import coordinax as cx
import unxt as u

q = {
    "x": u.Quantity(1.0, "km"),
    "y": u.Quantity(2.0, "km"),
    "z": u.Quantity(3.0, "km"),
}
v = {
    "x": u.Quantity(4.0, "km/s"),
    "y": u.Quantity(5.0, "km/s"),
    "z": u.Quantity(6.0, "km/s"),
}

qvec = cx.Vector(data=q, rep=cx.r.cart3d, role=cx.r.Pos())
vvec = cx.Vector(data=v, rep=cx.r.cart3d, role=cx.r.Vel())

q_sph = qvec.vconvert(cx.r.sph3d)
v_sph = vvec.vconvert(cx.r.sph3d, qvec)
```
