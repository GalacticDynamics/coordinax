# Working With Manifolds

This guide focuses on manifold functionality in `coordinax.manifolds`.

## Why Manifolds?

Charts describe coordinate systems. Manifolds describe **which charts are compatible for the same geometric space**.

In `coordinax`, a manifold owns an atlas:

- manifold: geometric object + validation entry point
- atlas: chart membership rules and default chart

If you only need raw chart transforms, use the charts guide. If you need compatibility guarantees, embeddings, or manifold-level workflows, use this guide.

## Atlas vs. Manifold

- A **manifold** owns an atlas of compatible charts.
- An **atlas** defines chart membership and defaults.
- Manifold-level coordinate methods validate chart compatibility before delegating to chart-level maps.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm

>>> M = cxm.EuclideanManifold(3)
>>> M.ndim
3
>>> M.default_chart()
Cart3D(M=Rn(3))

>>> M.has_chart(cxc.cart3d)
True
>>> M.has_chart(cxc.cart2d)
False
```

`M.has_chart(chart)` delegates to atlas compatibility checks. The manifold default chart comes from the atlas.

## Built-In Manifolds

### EuclideanManifold

`EuclideanManifold(n)` supports Euclidean charts of matching dimension.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm

>>> R2 = cxm.EuclideanManifold(2)
>>> R2.default_chart()
Cart2D(M=Rn(2))
>>> R2.has_chart(cxc.cart2d)
True
>>> R2.has_chart(cxc.polar2d)
True
```

### HyperSphericalManifold

`HyperSphericalManifold` supports intrinsic two-sphere charts and rejects planar Euclidean 2D charts.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm

>>> S2 = cxm.HyperSphericalManifold(2)
>>> S2.default_chart()
SphericalTwoSphere(M=Sn(2))
>>> S2.has_chart(cxc.sph2)
True
>>> S2.has_chart(cxc.cart2d)
False
```

## Guessing Manifolds

Use `guess_manifold` when you have data or a chart and need a manifold object.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm

>>> cxm.guess_manifold({"x": 1, "y": 2, "z": 3})
Rn(3)

>>> cxm.guess_manifold(cxc.sph2)
HyperSphericalManifold(ndim=2)
```

## Point Transitions

Use `cxc.pt_map` (or `cxm.pt_map`) to convert a point between two charts on the same manifold.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import unxt as u

>>> p = {"x": u.Q(1, "km"), "y": u.Q(1, "km")}
>>> p_pol = cxc.pt_map(p, cxc.cart2d, cxc.polar2d)
>>> sorted(p_pol)
['r', 'theta']
```

## Reading Metric Diagonals

Use `scale_factors` when you want the diagonal entries of the metric matrix in a chart.

This returns the metric diagonal $g_{ii}$, not the basis lengths $\sqrt{g_{ii}}$. The result is a 1-D `QMatrix` because different coordinate directions can carry different units.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import quaxed.numpy as jnp
>>> import unxt as u

>>> at = {
...     "r": u.Q(2, "km"),
...     "theta": u.Angle(jnp.pi / 2, "rad"),
...     "phi": u.Angle(0, "rad"),
... }

>>> gdiag = cxm.scale_factors(cxc.sph3d, at=at)
>>> gdiag.shape
(3,)
>>> jnp.allclose(gdiag.value, jnp.array([1.0, 4.0, 4.0]))
Array(True, dtype=bool)
>>> gdiag.unit.to_string()
'(, km2 / rad2, km2 / rad2)'
```

For generic metrics, `scale_factors` follows the metric matrix path and returns the diagonal. For `FlatMetric`, coordinax uses a more efficient specialization that avoids forming the full metric matrix.

## Measuring Angles Between Tangent Vectors

Use `angle_between` when you want the metric angle between two **tangent vectors** expressed as CDicts in a chart basis.

This is a tangent-space operation, not a point-to-point operation. The vectors are interpreted in the coordinate basis of the chosen chart, and the metric is evaluated at the base point `at`.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import quaxed.numpy as jnp
>>> import unxt as u

>>> at = {"x": u.Q(0, "m"), "y": u.Q(0, "m")}
>>> uvec = {"x": u.Q(1, "m"), "y": u.Q(0, "m")}
>>> vvec = {"x": u.Q(0, "m"), "y": u.Q(1, "m")}

>>> ang = cxm.angle_between(cxc.cart2d, uvec, vvec, at=at)
>>> jnp.allclose(u.ustrip("rad", ang), jnp.pi / 2)
Array(True, dtype=bool)
```

For curvilinear charts, the angle is still intrinsic, but the metric weights the coordinate directions at the supplied base point:

```{code-block} python
>>> metric = cxm.RoundMetric(ndim=2)
>>> at = {"theta": jnp.array(jnp.pi / 2), "phi": jnp.array(0.0)}
>>> uvec = {"theta": jnp.array(1.0), "phi": jnp.array(0.0)}
>>> vvec = {"theta": jnp.array(1.0), "phi": jnp.array(1.0)}

>>> ang = cxm.angle_between(metric, cxc.sph2, uvec, vvec, at=at)
>>> jnp.allclose(u.ustrip("rad", ang), jnp.pi / 4)
Array(True, dtype=bool)
```

If you have ordinary point coordinates rather than tangent components, first decide what tangent/displacement object you intend to compare. `angle_between` does not treat point-role CDicts as rays or displacements automatically.

## Embedding Workflows

Embeddings connect intrinsic manifold coordinates to ambient-space coordinates.

### Embeddings and Chart Transitions

The charts guide covers chart-to-chart point mapping on a single manifold: `pt_map` and `pt_map`. Embeddings introduce one additional primitive:

- embed map: carry a point from intrinsic to ambient coordinates ($\iota$)
- project map: recover intrinsic coordinates from ambient coordinates ($\pi$)

A realization map through an embedding composes as:

$$
\rho = \tau_M \circ \iota_{\text{coord}} \circ \tau_N,
$$

where $\tau_N$ and $\tau_M$ are exactly the chart-level operations from [Working With Charts](charts.md).

### EmbeddedChart (concise workflow)

Use `EmbeddedChart` for compact chart-facing embed/project operations.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import quaxed.numpy as jnp
>>> import unxt as u

>>> embedded = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(1, "km")))
>>> embedded.intrinsic
SphericalTwoSphere(M=Sn(2))
>>> embedded.ambient
Spherical3D(M=Rn(3))

>>> p_intrinsic = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
>>> p_ambient = cxm.pt_embed(p_intrinsic, embedded)
>>> sorted(p_ambient)
['phi', 'r', 'theta']

>>> p_back = cxm.pt_project(p_ambient, embedded)
>>> sorted(p_back)
['phi', 'theta']
>>> bool(jnp.allclose(u.ustrip("rad", p_back["theta"]), u.ustrip("rad", p_intrinsic["theta"])))
True

>>> p_cart = cxm.pt_map(p_ambient, embedded.ambient, cxc.cart3d)
>>> sorted(p_cart)
['x', 'y', 'z']
```

This workflow chains:

1. Intrinsic chart coordinates are converted by `pt_embed` (the cross-manifold step).
2. Ambient coordinates are then mapped via chart realization (standard chart transitions).

### EmbeddedManifold (explicit workflow)

Use `EmbeddedManifold` when you need explicit manifold objects with atlas compatibility checks.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import unxt as u

>>> em = cxm.EmbeddedManifold(
...     intrinsic=cxm.S2, ambient=cxm.R3,
...     embed_map=cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")),
... )
>>> em.ndim
2

>>> p_lonlat = {"lon": u.Q(30, "deg"), "lat": u.Q(10, "deg")}
>>> p_cart = cxm.pt_embed(p_lonlat, cxc.lonlat_sph2, cxc.cart3d, em)
>>> sorted(p_cart)
['x', 'y', 'z']
```

**When to use each:**

- `EmbeddedChart`: chart-centric workflows with minimal structure
- `EmbeddedManifold`: when manifold identity and atlas compatibility matter

### CustomEmbeddingMap (custom workflow)

Use `CustomEmbeddingMap` for user-defined embed/project rules while still reusing chart and manifold plumbing.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import unxt as u

>>> def embed_fn(p, /, *, usys=None):
...     return {"r": u.Q(5, "km"), "theta": p["theta"], "phi": p["phi"]}

>>> def project_fn(p, /, *, usys=None):
...     return {"theta": p["theta"], "phi": p["phi"]}

>>> custom_map = cxm.CustomEmbeddingMap(
...     intrinsic=cxc.sph2,
...     ambient=cxc.sph3d,
...     embed_fn=embed_fn,
...     project_fn=project_fn,
... )
>>> custom_embedded = cxm.EmbeddedChart(custom_map)

>>> p_intrinsic = {"theta": u.Q(1.2, "rad"), "phi": u.Q(0.3, "rad")}
>>> cxm.pt_embed(p_intrinsic, custom_embedded)["r"]
Q(5, 'km')
```

### Embeddings vs. Product Manifolds

Embeddings are asymmetric: an intrinsic manifold $N$ sits inside an ambient manifold $M$ via $\iota : N \hookrightarrow M$.

Product manifolds are symmetric: factors are independent peers and transitions are factorwise. Use `CartesianProductManifold` for independent combinations, not for submanifold embeddings.

## Advanced: Custom and Product Manifolds

### CustomAtlas + CustomManifold

When needed, build manifolds from explicit chart sets with `CustomAtlas` and `CustomManifold`.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm

>>> A = cxm.CustomAtlas(charts=(cxc.Cart2D, cxc.Polar2D), chart_default=cxc.cart2d)
>>> M = cxm.CustomManifold(A, metric=cxm.FlatMetric(2))

>>> M.has_chart(cxc.cart2d)
True
>>> M.has_chart(cxc.cart3d)
False
```

### CartesianProductManifold

Product manifolds combine independent factors and sum dimensions.

```{code-block} python
>>> import coordinax.manifolds as cxm

>>> MP = cxm.CartesianProductManifold(
...     factors=(cxm.S2, cxm.R1), factor_names=("S2", "R1")
... )
>>> MP.ndim
3
```

## Quick Reference

- Need compatibility checks around chart transitions: manifold methods
- Need manifold inference from data/charts: `guess_manifold`
- Need intrinsic-to-ambient conversion: `pt_embed` / `pt_project`
- Need custom chart membership rules: `CustomAtlas` + `CustomManifold`
- Need manifold products: `CartesianProductManifold`

:::{seealso}

[Working With Charts](charts.md)

[Manifolds API](../api/manifolds.md)

[Charts API](../api/charts.md)

:::
