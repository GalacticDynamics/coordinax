---
title: Metrics
---

# Metrics

Metrics define inner products and norms on tangent spaces. They determine how
physical components are interpreted and compared.

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

## Physical components vs coordinate derivatives

Physical components are the inputs to `physical_tangent_transform` and
`vconvert`. Coordinate derivatives can mix units and are not transformed by
these routines:

```
import unxt as u

qdot = {
    "r": u.Q(1.0, "km/s"),
    "theta": u.Q(1.0, "rad/s"),
    "phi": u.Q(2.0, "rad/s"),
}
```

## Metric objects

### Euclidean

$$
 g_{ij} = \delta_{ij}
$$

### Sphere (TwoSphere intrinsic)

$$
 g_{\theta\theta} = 1,\quad g_{\phi\phi} = \sin^2\theta
$$

### Minkowski (SpaceTimeCT)

$$
 g = \mathrm{diag}(-1, 1, 1, 1)
$$

## `metric_of` defaults

- Euclidean metric is the default for Euclidean reps exposed in
  `cx.charts.cart3d`.
- `TwoSphere` uses the intrinsic sphere metric.
- `SpaceTimeCT` uses Minkowski metric with signature `(-,+,+,+)`.
- `SpaceTimeEuclidean` uses Euclidean metric in 4D.

## Worked examples

Minkowski inner product (time-like negative):

```
import jax.numpy as jnp
import coordinax as cx
import unxt as u
from unxt.quantity import AllowValue

rep = cx.charts.SpaceTimeCT(cx.charts.cart3d)
eta = cx.metrics.metric_of(rep).metric_matrix(rep, {})
v = u.Q([2.0, 0.0, 0.0, 0.0], "km")
v_val = u.ustrip(AllowValue, "km", v)
bool(jnp.allclose(v_val @ eta @ v_val, -4.0))
```

Sphere metric at the equator:

```
import jax.numpy as jnp
import coordinax as cx
import unxt as u

p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
g = cx.metrics.metric_of(cx.charts.twosphere).metric_matrix(cx.charts.twosphere, p)
bool(jnp.allclose(g[1, 1], 1.0))
```

## Cross-links

- [Embedded manifolds guide](embedded_manifolds.md)
- `cx.charts.frame_cart`
- `cx.metrics.metric_of`
