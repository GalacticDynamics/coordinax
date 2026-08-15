# Working With Charts

This guide focuses only on chart functionality in `coordinax.charts`.

## What A Chart Is

- A **chart** defines coordinate component names and dimensions.
- Chart maps change coordinate representation while preserving the same point.
- Most charts are static descriptors; coordinate values live in dictionaries. A few charts carry parameters — see [Static And Parameterized Charts](#static-and-parameterized-charts).

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

Passing a component dictionary with `unxt.Quantity` values returns a `QuantityMatrix` (displayed as `QM`) whose element `[j, i]` carries the unit `output_unit_j / input_unit_i`:

```{code-block} python
>>> import coordinax.charts as cxc
>>> import unxt as u

>>> at = {"x": u.Q(1.0, "m"), "y": u.Q(0.0, "m"), "z": u.Q(0.0, "m")}
>>> J = cxc.jac_pt_map(at, cxc.cart3d, cxc.sph3d)
>>> J
QM(
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

All four components carry **length** units — the time coordinate is $ct$ — which is what keeps the metric dimensionless and lets a Lorentz boost be an ordinary dimensionless matrix. Because the metric is indefinite, `norm` and `geodesic_distance` refuse this chart; use `interval`, or `causal_character` and `proper_time` from `coordinax.manifolds.lorentzian`, instead. See the [Special Relativity tutorial](../tutorials/special_relativity.md).

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

## Static And Parameterized Charts

Every concrete chart is on exactly one of two branches; being on neither, or on both, is a `TypeError` at class-creation time.

- `AbstractStaticChart` — no parameters. Concrete subclasses register themselves with JAX as static, so they have zero pytree leaves and are always hashable. Staticness is structural, inherited from the branch, not a decorator that can be forgotten.
- `AbstractParameterizedChart` — carries parameters. It is an `equinox.Module`, and therefore a pytree.

```{code-block} python
>>> import coordinax.charts as cxc

>>> isinstance(cxc.cart3d, cxc.AbstractStaticChart)
True
>>> issubclass(cxc.ProlateSpheroidal3D, cxc.AbstractParameterizedChart)
True
```

### Opt-In Differentiability

A parameterized chart only has leaves if it is given something with leaves. `ProlateSpheroidal3D` accepts its focal length `Delta` as either quantity type, so differentiability is chosen per instance:

```{code-block} python
>>> import jax
>>> import unxt as u

>>> static = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
>>> len(jax.tree.leaves(static))
0

>>> dynamic = cxc.ProlateSpheroidal3D(Delta=u.Q(2.0, "m"))
>>> len(jax.tree.leaves(dynamic))
1
```

A `StaticQuantity` behaves exactly as charts always have: hashable, no leaves, baked into the `jit` cache key. A `Quantity` is a live array, so it can be differentiated through and `jit` traces once across many values of it:

```{code-block} python
>>> at = {"mu": u.Q(12.0, "m2"), "nu": u.Q(0.5, "m2"), "phi": u.Q(0.3, "rad")}

>>> def x_of(chart):
...     return cxc.pt_map(at, chart, cxc.cart3d)["x"].ustrip("m")

>>> jax.grad(x_of)(dynamic).Delta
Q(-0.45135407, 'm')
```

Both of these apply to a chart **passed as an argument to, or built inside, a traced function**. They do not extend to a chart carried inside a `coordinax.Point`: a point stores its chart in the pytree _structure_, not among its children, so it has the same three leaves either way.

```{code-block} python
>>> import coordinax as cx

>>> len(jax.tree.leaves(cx.Point(at, static)))
3
>>> len(jax.tree.leaves(cx.Point(at, dynamic)))
3
```

The cost is caching, not gradients: since the chart is structural data, `jit` over points whose charts hold a dynamic `Delta` retraces on every call — the equality rule below guarantees a cache miss. Pass the chart as its own argument when you want one trace across many `Delta`.

### Charts With Dynamic Parameters Compare Unequal

```{code-block} python
>>> a = cxc.ProlateSpheroidal3D(Delta=u.Q(2.0, "m"))
>>> b = cxc.ProlateSpheroidal3D(Delta=u.Q(2.0, "m"))
>>> a == b
False
>>> a == a
True
```

Equality is deliberately conservative: two charts with dynamic parameters are equal only when they are the same object, even when numerically identical. Under `jit` those parameters are tracers with no values to compare, so any rule that inspected them would answer differently inside and outside `jit`. Hashing ignores dynamic fields for the same reason, which keeps equal charts hashing equal. Static parameters are compared by value, as before:

```{code-block} python
>>> s1 = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
>>> s2 = cxc.ProlateSpheroidal3D(Delta=u.StaticQuantity(2.0, "m"))
>>> s1 == s2, hash(s1) == hash(s2)
(True, True)
```

### Static Charts Reject Arrays

Registering a chart as static makes the whole instance one static node, so an array held in a field would report zero leaves and be silently baked in as a constant. Static charts refuse this at construction:

```{code-block} python
>>> cxc.GalileanCT(cxc.ProlateSpheroidal3D(Delta=u.Q(2.0, "m")))
Traceback (most recent call last):
    ...
TypeError: GalileanCT is a static chart, but ['spatial_chart'] hold arrays. ...
```

The guard walks pytree leaves, so it only sees an array that is _in_ the pytree: one buried inside a value that is itself registered static would be a single opaque leaf and slip through. The way out is to put the parameter on the parameterized branch, not to make the guard stricter. `EmbeddedChart` is the worked example — `TwoSphereIn3D.radius` is a coordinate value that `pt_embed` propagates into output points, so forcing it static would break every embedded coordinate; instead the embedding map is a pytree and the chart is parameterized:

```{code-block} python
>>> import coordinax.manifolds as cxm

>>> len(jax.tree.leaves(cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.Q(2.0, "km")))))
1
>>> len(jax.tree.leaves(cxm.EmbeddedChart(cxm.TwoSphereIn3D(radius=u.StaticQuantity(2.0, "km")))))
0
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
