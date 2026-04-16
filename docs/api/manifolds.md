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

# Manifold-level chart transition map.
p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
p_sph = M.pt_map(p, cxc.cart3d, cxc.sph3d)

# Guess manifold from data/chart.
M2 = cxm.guess_manifold(p)
M3 = cxm.guess_manifold(cxc.sph2)

# Metric angle between two tangent vectors.
at = {"x": u.Q(0, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
uvec = {"x": u.Q(1, "km"), "y": u.Q(0, "km"), "z": u.Q(0, "km")}
vvec = {"x": u.Q(0, "km"), "y": u.Q(1, "km"), "z": u.Q(0, "km")}
ang = M.angle_between(cxc.cart3d, uvec, vvec, at=at)
```

## Functional API

- `guess_manifold`: infer a manifold from manifold/chart/data inputs
- `scale_factors`: return the metric diagonal in a chart at a base point
- `angle_between`: return the metric angle between two tangent-vector CDicts
- `pt_embed`: embed intrinsic coordinates into ambient coordinates
- `pt_project`: project ambient coordinates back to intrinsic chart coordinates
- `pt_map`: manifold-related re-export of point realization map

## Available Objects

### Manifolds

- `AbstractManifold`: base manifold interface
- `EuclideanManifold` / `euclidean3d`: Euclidean manifold family and 3D convenience
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

- Manifold methods (for example `pt_map`) validate atlas compatibility before delegating to chart-level coordinate rules.
- For intrinsic two-sphere workflows, use `HyperSphericalManifold` and intrinsic two-sphere charts (`sph2`, `lonlat_sph2`, etc.) rather than Euclidean 2D charts.

```{eval-rst}

.. currentmodule:: coordinax.manifolds

.. automodule:: coordinax.manifolds
    :exclude-members: aval, default, materialise, enable_materialise

```
