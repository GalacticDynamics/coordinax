# `coordinax.manifolds`

The `coordinax.manifolds` module provides manifold and atlas objects, plus manifold-level point operations.

## Overview

In `coordinax`, a manifold is represented as a pair $(M, \mathcal{A})$:

- $M$: the geometric manifold
- $\mathcal{A}$: an atlas describing compatible charts

Manifold objects are responsible for compatibility checks (which charts belong on the manifold) and for manifold-level wrappers around chart operations.

For a step-by-step walkthrough, see [Working With Manifolds](../guides/manifolds.md).

## Quick Start

```python
import coordinax.charts as cxc
import coordinax.manifolds as cxm
import unxt as u

# Euclidean manifold in 3 dimensions.
M = cxm.EuclideanManifold(3)

# Check chart compatibility.
assert M.has_chart(cxc.cart3d)
assert not M.has_chart(cxc.cart2d)

# Chart-level point transition map.
p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
p_sph = cxc.pt_map(p, cxc.cart3d, cxc.sph3d)

# Guess manifold from data/chart.
M2 = cxm.guess_manifold(p)
M3 = cxm.guess_manifold(cxc.sph2)

# Metric angle between two tangent vectors.
at = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
uvec = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
vvec = {"x": u.Q(0, "km"), "y": u.Q(1, "km"), "z": u.Q(0, "km")}
ang = cxm.angle_between(cxc.cart3d, uvec, vvec, at=at)
```

## Functional API

- `guess_manifold`: infer a manifold from manifold/chart/data inputs
- `scale_factors`: return the metric diagonal in a chart at a base point
- `angle_between`: return the metric angle between two tangent-vector CDicts
- `norm`: compute the Riemannian norm $\|v\|_g = \sqrt{g_p(v,v)}$ of a tangent vector in a chart. Requires a **positive-definite** metric; raises `NotImplementedError` for an indefinite one (e.g. Minkowski), where the square root would be `nan`
- `geodesic_distance`: length of the shortest path between two points _along the manifold_. Symmetric and chart-invariant, computed from the manifold's geometry: the straight line in flat space, the great circle on a sphere. A manifold with no closed-form geodesic raises `NotImplementedError` rather than approximating, as does Minkowski, whose indefinite metric admits no distance
- `interval`: signed squared interval $\Delta s^2 = \Delta x^\top G\,\Delta x$ of the _coordinate difference_, with the metric taken at the first point. Defined for **every** metric, including indefinite ones, and the causal invariant when Lorentzian. It is not the squared `geodesic_distance` except where the metric is constant along the path, i.e. on a flat manifold in Cartesian coordinates -- flatness alone is not enough, since a curvilinear chart on flat space has a varying metric. The verbs that read its sign need a timelike direction and live in the `coordinax.manifolds.lorentzian` sub-namespace below
- `pt_embed`: embed intrinsic coordinates into ambient coordinates
- `pt_project`: project ambient coordinates back to intrinsic chart coordinates
- `pt_map`: manifold-related re-export of point realization map

## Available Objects

### Manifolds

- `AbstractLorentzianMetricField`: structural marker for metrics with a Lorentzian signature; gates the `lorentzian` sub-namespace below
- `AbstractManifold`: base manifold interface
- `EuclideanManifold` / `R3`: Euclidean manifold family and 3D convenience
- `HyperSphericalManifold`: intrinsic two-sphere manifold
- `CartesianProductManifold`: Cartesian product manifold
- `EmbeddedManifold`: manifold with explicit embedding into an ambient manifold
- `CustomManifold`: manifold backed by a caller-provided atlas

### Atlases

- `AbstractAtlas`: base atlas interface
- `EuclideanAtlas`: atlas for Euclidean charts of fixed dimension
- `HyperSphericalAtlas`: atlas for intrinsic two-sphere charts
- `CartesianProductAtlas`: atlas for product manifolds
- `CustomAtlas`: explicit atlas with caller-controlled chart membership

### Embeddings and Embedded Charts

- `AbstractEmbeddingMap`: base embedding map interface
- `CustomEmbeddingMap`: user-defined embedding maps
- `TwoSphereIn3D` / `embedded_twosphere`: standard two-sphere embedding in 3D
- `EmbeddedChart`: convenience chart wrapper combining intrinsic chart and embedding

## Notes

- Manifold methods delegate chart transitions to `cxc.pt_map`.
- For intrinsic two-sphere workflows, use `HyperSphericalManifold` and intrinsic two-sphere charts (`sph2`, `lonlat_sph2`, etc.) rather than Euclidean 2D charts.

```{eval-rst}

.. currentmodule:: coordinax.manifolds

.. automodule:: coordinax.manifolds
    :exclude-members: aval, default, materialise, enable_materialise

```

## `coordinax.manifolds.lorentzian`

A sub-namespace for measurements that need a **timelike direction** — gated on the metric type `AbstractLorentzianMetricField` (signature $(-,+,\ldots,+)$), not on a chart. A metric without one has no method here, rather than a method that accepts and then refuses.

- `causal_character`: classify a pair of events as `"timelike"`, `"null"`, or `"spacelike"`. Returns a `str`, so not `jit`-able; branch on the sign of `interval` inside a trace
- `proper_time`: elapsed proper time between two timelike-separated events
- `proper_distance`: proper distance between two spacelike-separated events
- `rapidity_between`: relative rapidity between two timelike tangent vectors — the hyperbolic counterpart of `angle_between`, which refuses that pair
- `interval`: **re-exported** for convenience — canonical in `coordinax.manifolds`, since the signed quadratic form is defined for _every_ metric. It appears here because `causal_character` is its sign and `proper_time` its root

Named for the signature rather than for "spacetime" deliberately: `charts.galileanct` is a 4-D Galilean spacetime and is **not** Lorentzian, so a `spacetime` namespace would promise membership these verbs refuse. Named `lorentzian` rather than `minkowski` because the gate is the signature — a curved spacetime metric (Schwarzschild, FLRW) inherits the marker and acquires all of them.

```pycon
>>> import unxt as u
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm

>>> birth = {k: u.Q(0.0, "m") for k in ("ct", "x", "y", "z")}
>>> death = {
...     "ct": u.Q(5.0, "m"),
...     "x": u.Q(1.0, "m"),
...     "y": u.Q(0.0, "m"),
...     "z": u.Q(0.0, "m"),
... }

>>> cxm.lorentzian.causal_character(cxc.minkowskict, birth, death)
'timelike'

>>> cxm.lorentzian.proper_time(cxc.minkowskict, birth, death).uconvert("ns").round(2)
Q(16.34, 'ns')
```
