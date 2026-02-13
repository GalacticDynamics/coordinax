---
title: Embedded Manifolds
---

# Embedded Manifolds

This guide describes how to represent charts on embedded manifolds and how
physical components are pushed forward and projected using orthonormal frames.

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

## Embedded manifold model

An embedded manifold uses an intrinsic chart for coordinates and an ambient
representation for the embedding.

Mathematically:

$$
\iota: U \subset M \to \mathbb{R}^n, \qquad q \mapsto x(q)
$$

In `coordinax`, this is represented by `cxe.EmbeddedManifold` with `chart_kind`
(intrinsic) and `ambient_kind` (ambient).

## TwoSphere embedding

For the unit two-sphere embedded in Cartesian 3D, the embedding is:

$$
x = R \sin\theta \cos\phi,\quad
y = R \sin\theta \sin\phi,\quad
z = R \cos\theta
$$

where `R = params["R"]` supplies the length scale.

`TwoSphere` has no global Cartesian 2D rep; use `EmbeddedManifold` for ambient
Cartesian coordinates.

### Tangent orthonormal frame

The orthonormal tangent frame (columns of `frame_cart`) is:

$$
\hat e_\theta = (\cos\theta\cos\phi,\; \cos\theta\sin\phi,\; -\sin\theta),\quad
\hat e_\phi = (-\sin\phi,\; \cos\phi,\; 0)
$$

These vectors are orthonormal with respect to the ambient Euclidean metric.

## Position embedding and projection

`embed_point` maps intrinsic coordinates to ambient coordinates. `project_point`
recovers intrinsic coordinates from ambient coordinates.

Projection uses:

$$
\theta = \arccos\left(\frac{z}{r}\right),\quad
\phi = \operatorname{atan2}(y, x),\quad
r = \sqrt{x^2+y^2+z^2}
$$

At the poles (`sin(theta)=0`), longitude is not unique; `project_point` sets
`phi = 0` by convention.

Worked example (radius check):

```
import jax.numpy as jnp
import coordinax as cx
import unxt as u

rep = cxe.EmbeddedManifold(
    chart_kind=cxc.twosphere,
    ambient_kind=cxc.cart3d,
    params={"R": u.Q(2.0, "km")},
)

p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
q = cxe.embed_point(rep, p)
r2 = (
    u.uconvert("km", q["x"]) ** 2
    + u.uconvert("km", q["y"]) ** 2
    + u.uconvert("km", q["z"]) ** 2
)
bool(jnp.allclose(r2.value, 4.0))
```

## Physical tangent vectors

`embed_tangent` and `project_tangent` operate on physical components (not
coordinate derivatives) using the orthonormal frame from `frame_cart`:

$$
v_{\text{cart}} = B(q)\,v_{\text{rep}},\quad
v_{\text{rep}} = B(q)^{\mathsf{T}} v_{\text{cart}}
$$

Worked example (round-trip):

```
import jax.numpy as jnp
import coordinax as cx
import unxt as u

rep = cxe.EmbeddedManifold(
    chart_kind=cxc.twosphere,
    ambient_kind=cxc.cart3d,
    params={"R": u.Q(1.0, "km")},
)

p = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
v = {"theta": u.Q(1.0, "km/s"), "phi": u.Q(0.0, "km/s")}

v_cart = cxe.embed_tangent(rep, v, at=p)
v_back = cxe.project_tangent(rep, v_cart, at=p)
bool(jnp.allclose(u.uconvert("km/s", v_back["theta"]).value, 1.0))
```

## Metrics and frames

`frame_cart` provides the orthonormal frame in ambient Cartesian components.
`metric_of` resolves the active metric (Euclidean, Sphere, or Minkowski).

See also:

- [Metrics guide](metrics.md)
- `cxc.frame_cart`
- `cxm.metric_of`
