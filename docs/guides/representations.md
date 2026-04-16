# Working With Representations

This guide focuses on representation functionality in `coordinax.representations` and how it connects `coordinax.charts` and `coordinax.manifolds` workflows.

## Why Representations?

`coordinax` separates three concerns:

- **Charts**: how coordinates are written (`x, y, z`, `r, theta, phi`, ...)
- **Manifolds**: which charts are compatible for the same geometric space
- **Representations**: what kind of geometric object the component data means

Representations are the bridge layer. They let conversion APIs choose the right transformation law while staying independent of chart choice.

In the current implementation, point data is the built-in representation flow.

## The Representation Triple

A representation is a triple $R = (K, B, S)$:

- $K$: geometry kind (`AbstractGeometry`), for example `PointGeometry`
- $B$: basis kind (`AbstractBasis`), for point data this is `NoBasis`
- $S$: semantic kind (`AbstractSemanticKind`), for point data this is `Location`

```{code-block} python
>>> import coordinax.representations as cxr

>>> rep = cxr.Representation(cxr.PointGeometry(), cxr.NoBasis(), cxr.Location())
>>> rep
Representation(geom_kind=PointGeometry(), basis=NoBasis(), semantic_kind=Location())

>>> rep == cxr.point
True
```

Canonical point representation:

$$
(\mathrm{PointGeometry},\, \mathrm{NoBasis},\, \mathrm{Location}).
$$

`NoBasis` does **not** mean "no coordinates". It means basis choice is not part of affine point representation semantics.

## How This Ties Charts And Manifolds Together

For point data, the stack is:

1. Chart-level point maps define coordinate-change mechanics.
2. Manifold-level wrappers enforce chart compatibility for a given manifold.
3. Representation-level conversion (`cconvert`) dispatches through the point realization/transition machinery while preserving representation meaning.

This means you can write representation-aware code without duplicating chart logic, and still rely on manifold checks where needed.

## Transition Layer vs. Representation Layer

Use each API by intent:

- `coordinax.charts.pt_map`: same-manifold chart transition
- `coordinax.charts.pt_map`: general point map (including realization-style paths)
- `AbstractManifold.pt_map`: same as transition map plus atlas compatibility checks
- `coordinax.representations.cconvert`: representation-aware top-level conversion API
- `coordinax.representations.cmap`: reusable partial conversion map

## End-To-End Workflow

This example shows one point represented across chart, manifold, and representation layers.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import coordinax.representations as cxr
>>> import unxt as u

>>> p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}

>>> # 1) Chart-level transition map
>>> q_chart = cxc.pt_map(p, cxc.cart3d, cxc.sph3d)
>>> sorted(q_chart)
['phi', 'r', 'theta']

>>> # 2) Manifold-level transition map (adds atlas compatibility checks)
>>> M = cxm.EuclideanManifold(3)
>>> q_mfld = M.pt_map(p, cxc.cart3d, cxc.sph3d)
>>> q_mfld == q_chart
True

>>> # 3) Representation-aware conversion for point data
>>> q_rep = cxr.cconvert(p, cxc.cart3d, cxr.point, cxc.sph3d, cxr.point)
>>> q_rep == q_chart
True
```

## Reusable Representation Maps

Use `cmap` when you repeatedly apply the same conversion pattern.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.representations as cxr
>>> import unxt as u

>>> to_sph = cxr.cmap(cxc.cart3d, cxr.point, cxc.sph3d)
>>> p = {"x": u.Q(1, "m"), "y": u.Q(0, "m"), "z": u.Q(0, "m")}
>>> to_sph(p)
{'r': Q(1., 'm'), 'theta': Q(1.57079633, 'rad'), 'phi': Q(0., 'rad')}
```

## Realization Context (Intrinsic vs. Ambient)

When moving between intrinsic and ambient descriptions, use realization-style chart maps. This is where charts and manifolds meet most directly.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import coordinax.manifolds as cxm
>>> import unxt as u

>>> embedded = cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(1, "km")))

>>> p_intrinsic = {"theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
>>> p_ambient = cxm.pt_embed(p_intrinsic, embedded)
>>> sorted(p_ambient)
['phi', 'r', 'theta']

>>> p_cart = cxc.pt_map(p_ambient, embedded.ambient, cxc.cart3d)
>>> sorted(p_cart)
['x', 'y', 'z']
```

For point data, representation-aware conversion uses this same realization machinery under the hood, with representation checks.

## Current Scope And Future Directions

Current built-in representation conversions are point-first:

- `PointGeometry`
- `NoBasis`
- `Location`

The representation design is intentionally extensible. Future geometric kinds (for example tangent and cotangent objects) can use different transformation categories (such as Jacobian pushforward/pullback) while keeping the same chart and manifold interfaces.

## Quick Reference

- Need only a same-manifold coordinate rewrite: `pt_map`
- Need general point mapping behavior: `pt_map`
- Need manifold compatibility checks: manifold methods like `M.pt_map`
- Need representation-aware conversions: `cconvert`
- Need reusable conversion callables: `cmap`
- Need to infer basis kind from data: `guess_basis_kind`
- Need to infer semantic kind from data: `guess_semantic_kind`

:::{seealso}

[Representations API](../api/representations.md)

[Working With Charts](charts.md)

[Working With Manifolds](manifolds.md)

:::
