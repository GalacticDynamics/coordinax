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

## Physical components vs coordinate derivatives

Physical components are the inputs to `diff_map` and `vconvert`. Coordinate
derivatives can mix units and are not transformed by these routines:

```python
import unxt as u

qdot = {
    "r": u.Quantity(1.0, "km/s"),
    "theta": u.Quantity(1.0, "rad/s"),
    "phi": u.Quantity(2.0, "rad/s"),
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

- Euclidean metric is the default for Euclidean reps exposed in `cx.r`.
- `TwoSphere` uses the intrinsic sphere metric.
- `SpaceTimeCT` uses Minkowski metric with signature `(-,+,+,+)`.
- `SpaceTimeEuclidean` uses Euclidean metric in 4D.

## Worked examples

Minkowski inner product (time-like negative):

```python
import jax.numpy as jnp
import coordinax as cx
import unxt as u
from unxt.quantity import AllowValue

rep = cx.r.SpaceTimeCT(cx.r.cart3d)
eta = cx.r.metric_of(rep).metric_matrix(rep, {})
v = u.Quantity([2.0, 0.0, 0.0, 0.0], "km")
v_val = u.ustrip(AllowValue, "km", v)
bool(jnp.allclose(v_val @ eta @ v_val, -4.0))
```

Sphere metric at the equator:

```python
import jax.numpy as jnp
import coordinax as cx
import unxt as u

p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
g = cx.r.metric_of(cx.r.twosphere).metric_matrix(cx.r.twosphere, p)
bool(jnp.allclose(g[1, 1], 1.0))
```

## Cross-links

- [Embedded manifolds guide](embedded_manifolds.md)
- `cx.r.frame_to_cart`
- `cx.r.metric_of`
