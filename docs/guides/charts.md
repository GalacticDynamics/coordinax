# Working With Charts

This guide focuses only on chart functionality in `coordinax.charts`.

## What A Chart Is

- A **chart** defines coordinate component names and dimensions.
- Chart maps change coordinate representation while preserving the same point.
- Charts are static descriptors; coordinate values live in dictionaries.

```{code-block} python
>>> import coordinax.charts as cxc

>>> cxc.cart3d.components
('x', 'y', 'z')

>>> cxc.sph3d.coord_dimensions
('length', 'angle', 'angle')
```

## Choosing The Right Map

Use these chart APIs by intent:

- `pt_map`: point coordinate change on the same manifold
- `pt_map`: general point map interface
- `cartesian_chart`: chart selection only (no coordinate data transformation)

For same-manifold chart changes, transition and realization maps agree:

Use `pt_map` for same-manifold chart transitions and `pt_map` for the general point map interface.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> p = {"x": u.Q(1, "km"), "y": u.Q(2, "km"), "z": u.Q(3, "km")}
>>> p_sph = cxc.pt_map(p, cxc.cart3d, cxc.sph3d)
>>> sorted(p_sph)
['phi', 'r', 'theta']

>>> p_sph2 = cxc.pt_map(p, cxc.cart3d, cxc.sph3d)
>>> p_sph2 == p_sph
True
```

Chart selection is independent of point data:

```{code-block} python
>>> import coordinax.charts as cxc

>>> cxc.cartesian_chart(cxc.sph3d)
Cart3D(M=Rn(3))
```

## Inferring Charts And Normalizing Inputs

`guess_chart` infers a chart from keys or array shape heuristics. `cdict` normalizes different input forms to component dictionaries.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> cxc.guess_chart({"x": 1.0, "y": 2.0, "z": 3.0})
Cart3D(M=Rn(3))

>>> cxc.guess_chart(frozenset(("x", "y", "z")))
Cart3D(M=Rn(3))

>>> q = u.Q([1.0, 2.0, 3.0], "m")
>>> cxc.cdict(q)
{'x': Q(1., 'm'), 'y': Q(2., 'm'), 'z': Q(3., 'm')}
```

`guess_chart` caveats:

- Key-based inference uses component-name sets, so it is not a unique identifier when multiple chart types share names.
- Array/quantity shape inference is limited to trailing axis sizes 1, 2, or 3.

## Product Charts

Product-chart transitions are factorwise.

```{code-block} python
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> st_cart = cxc.CartesianProductChart((cxc.time1d, cxc.cart3d), ("ct", "q"))
>>> st_sph = cxc.CartesianProductChart((cxc.time1d, cxc.sph3d), ("ct", "q"))

>>> p_st = {"ct.t": u.Q(1, "km"), "q.x": u.Q(2, "km"), "q.y": u.Q(0, "km"), "q.z": u.Q(0, "km")}
>>> q_st = cxc.pt_map(p_st, st_cart, st_sph)
>>> sorted(q_st)
['ct.t', 'q.phi', 'q.r', 'q.theta']

>>> prod = cxc.CartesianProductChart((cxc.time1d, cxc.sph3d), ("t", "q"))
>>> prod.components
('t.t', 'q.r', 'q.theta', 'q.phi')
```

## Computing Jacobians

`jac_pt_map` returns the coordinate-transformation Jacobian $J^j{}_i = \partial \phi^j / \partial q^i$ evaluated at a base point, where $\phi$ is the transition function from `from_chart` to `to_chart`.

### Direct call — quantity-valued dictionary input

Passing a component dictionary with `unxt.Quantity` values returns a `QMatrix` whose element `[j, i]` carries the unit `output_unit_j / input_unit_i`:

```{code-block} python
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
>>> J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
>>> J
QMatrix(
    [[ 1.,  0.,  0.],
     [-0., -0., -1.],
     [ 0.,  1.,  0.]],
    '((, , ), (rad / m, rad / m, rad / m), (rad / m, rad / m, rad / m))'
)
>>> J.shape
(3, 3)
```

### Plain-array input with a unit system

Pass a plain numeric dict and supply `usys` to interpret the dimensionless elements:

```{code-block} python
>>> import jax.numpy as jnp
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> at_arr = {"x": jnp.array(1.0), "y": jnp.array(0.0), "z": jnp.array(0.0)}
>>> J2 = cxc.jac_pt_map(at_arr, cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
>>> J2.shape
(3, 3)
```

### Curried form — efficient reuse across many points

`jac_pt_map(None, *args, **kwargs)` -- or more explicitly, `jac_pt_map(from_chart, to_chart, usys=usys)` -- returns a callable that can be applied to many points without re-building the underlying point-map partial each time. This is the recommended pattern for use inside `jax.jit` and `jax.vmap`:

```{code-block} python
>>> import jax
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> jac_fn = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)

>>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
>>> J = jac_fn(at)
>>> J.shape
(3, 3)
```

### JIT and vmap compatibility

The curried form is JIT- and vmap-compatible:

```{code-block} python
>>> import jax
>>> import jax.numpy as jnp
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> jac_fn = cxc.jac_pt_map(cxc.cart3d, cxc.sph3d, usys=u.unitsystems.si)
>>> jac_jit = jax.jit(jac_fn)

>>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
>>> jac_jit(at).shape
(3, 3)

>>> jac_vmap = jax.vmap(jac_jit)
>>> at_batch = jax.tree.map(lambda x: x[None], at)  # Add batch dimension
>>> jac_vmap(at_batch).shape
(1, 3, 3)
```

### Chain rule via Jacobian composition

The coordinate-change chain rule states that composing two Jacobians gives the Jacobian of the composed map. Use `quaxed.numpy.matmul` (or `coordinax._src.quantity_matrix.qnp.matmul`) for unit-aware matrix multiply:

```{code-block} python
>>> import quaxed.numpy as qnp
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> at_cart = {"x": u.Q(1.0, "m"), "y": u.Q(0.5, "m"), "z": u.Q(2.0, "m")}

>>> J_cs = cxc.jac_pt_map(at_cart, cxc.cart3d, cxc.sph3d)

>>> # Transform point to intermediate chart
>>> at_sph = cxc.pt_map(at_cart, cxc.cart3d, cxc.sph3d)
>>> J_sc = cxc.jac_pt_map(at_sph, cxc.sph3d, cxc.cart3d)

>>> # Chain rule: identity (up to floating-point)
>>> J_composed = qnp.matmul(J_sc, J_cs)
>>> J_composed.shape
(3, 3)
```

## Spacetime Charts

`coordinax.charts` provides two 4D spacetime chart types.

**Minkowski spacetime** (`minkowskict`) is a flat, fixed-component chart with signature $(-,+,+,+)$:

```{code-block} python
>>> import coordinax.charts as cxc
>>> cxc.minkowskict.components
('ct', 'x', 'y', 'z')
>>> cxc.minkowskict.cartesian is cxc.minkowskict
True
```

**Galilean spacetime** (`galileanct`) is a parametric product chart `time1d × spatial_chart`. The default spatial chart is `cart3d`, giving components `(ct, x, y, z)`:

```{code-block} python
>>> import coordinax.charts as cxc
>>> cxc.galileanct.components
('ct', 'x', 'y', 'z')
>>> cxc.galileanct.spatial_chart
Cart3D(M=Rn(3))
```

The spatial factor can be changed at construction time. Chart conversions on the spatial part work factorwise — the `ct` component is untouched:

```{code-block} python
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> st_sph = cxc.GalileanCT(cxc.sph3d)
>>> st_sph.components
('ct', 'r', 'theta', 'phi')

>>> p = {"ct": u.Q(0.0, "km"), "r": u.Q(1.0, "km"), "theta": u.Q(1.0, "rad"), "phi": u.Q(0.5, "rad")}
>>> p_cart = cxc.pt_map(p, st_sph, cxc.galileanct)
>>> sorted(p_cart)
['ct', 'x', 'y', 'z']
```

`galileanct.cartesian` returns `self` when the spatial chart is already Cartesian; for a non-Cartesian variant it returns a new `GalileanCT` with a Cartesian spatial chart:

```{code-block} python
>>> cxc.galileanct.cartesian is cxc.galileanct
True
>>> cxc.GalileanCT(cxc.sph3d).cartesian == cxc.galileanct
True
```

## Quick Reference

- If you already know the source and target charts: `pt_map`
- If you are writing general chart-to-chart point code: `pt_map`
- If you need a canonical Cartesian chart object: `cartesian_chart`
- If your input type varies (dict/quantity/array): `cdict` and `guess_chart`

:::{seealso}

[Charts API](../api/charts.md)

[Working With Manifolds](manifolds.md)

:::
