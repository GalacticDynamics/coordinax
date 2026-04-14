---
jupytext:
  formats: md:myst
kernelspec:
  display_name: "Python 3"
  language: "python"
  name: "python3"
---

# Performance Optimization with Unitful Quantities

In this guide, we'll explore how to think about performance optimization when working with `coordinax` objects in JAX. The key insight is understanding **where the overhead lives** and **when it matters**.

## Key Concepts

1. **Wrapper overhead**: Operations on objects have overhead compared to raw JAX arrays.
2. **JIT removes overhead**: JAX's JIT compiler can eliminate much of this wrapper overhead by tracing through the code.
3. **Pytree complexity**: Objects are JAX pytrees, which adds cost when crossing JIT boundaries (converting between traced and non-traced values).
4. **Strategy**: The secret to performance is to **minimize pytree conversions at the boundary** between traced and non-traced code.

Let's explore this last point a little more. The details will become clearer in the examples below, but the general idea is that for optimal performance you want to structure your code so that the JIT-compiled functions take raw arrays as input and output, and you convert to/from `coordinax` objects inside the JIT-compiled function. This way, the overhead of working with `coordinax` objects is paid only once per JIT compilation, rather than on every call. Also as a note, `coordinax` is not "special" in this regard -- any PyTree object will have this same overhead at the boundary, so this is a general principle for working with JAX and PyTrees effectively.

The pseudo-code below illustrates the idea:

```text
@jax.jit
def function_that_takes_pytrees(*objects: PyTree):
    # This function can take and return PyTrees,
    # but it will be slower due to wrapper overhead and pytree complexity.
    ...

@jax.jit
def optimized_function(*arrays: Array):
    # This function takes and returns raw arrays,
    # so it can be optimized by JIT without overhead.
    # Inside this function, we can convert to/from PyTrees as needed,
    # but this conversion will be compiled away by JIT.
    ...
```

## Coordinate Changes

Let's start by importing the libraries we'll need and setting up some test data.

```{code-cell} ipython3
import functools as ft
from dataclasses import replace

from jaxtyping import Array
import jax
import jax.numpy as jnp
import quax

from jaxmore import vmap
import unxt as u

import coordinax.main as cx
```

We'll define this function now:

```{code-cell} ipython3
usys = u.unitsystems.si

_c2s_cx = cx.pt_map(cx.cart3d, cx.sph3d, usys=usys)
c2s_cx = jax.jit(vmap(_c2s_cx))
```

```{note}
This one `coordinax`-backed function will be able to transform ALL of the object types we work with without modification, thanks to the way `coordinax` objects are designed to work with JAX and Quax.
```

### Array

#### Basic JAX

```{code-cell} ipython3
@jax.jit
@vmap
def c2s_arr(x: Array, /) -> Array:
    r = jnp.linalg.norm(x, axis=-1)
    theta = jnp.acos(x[..., 2] / r)
    phi = jnp.arctan2(x[..., 1], x[..., 0])
    return jnp.stack((r, theta, phi), axis=-1)

xarr = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

%time jax.block_until_ready(c2s_arr(xarr))
%timeit jax.block_until_ready(c2s_arr(xarr))

c2s_arr(xarr)
```

This is a good baseline, but it's applied to a raw JAX array, which in `coordinax` is assumed to be in Cartesian coordinates. `coordinax` allows us to work with coordinates in many different types of charts. Let's see how the performance compares when we use `coordinax` objects, especially after JIT compilation.

#### Coordinax

Now let's work with `coordinax` and perform the same coordinate change. We'll see how the performance compares, especially after JIT compilation.

The key to performance is to **close over the static quantities** (like the charts and unit system) so that JAX can optimize the computation effectively. This means we want to avoid passing `coordinax` objects directly into JIT-compiled functions if they contain static information that can't be traced.

Let's time this function with raw arrays first, to see the baseline overhead of using `coordinax` objects.

```{code-cell} ipython3
%time jax.block_until_ready(c2s_cx(xarr))
%timeit jax.block_until_ready(c2s_cx(xarr))

c2s_cx(xarr)
```

There is NONE! The performance is the same as the hardcoded version. This is because we closed over the static arguments (the charts and unit system) so that JAX can optimize the computation effectively. The `coordinax` objects are only used inside the JIT-compiled function, so there is no overhead from pytrees or wrappers at the boundary.

Let's see what happens if we don't close over the static arguments:

```{code-cell} ipython3
c2s_bad = jax.jit(
  vmap(cx.pt_map, in_axes=(0, None, None), in_kw={"usys": None}),
  static_argnums=(1, 2), static_argnames=("usys"),
)

usys = u.unitsystems.si

%time jax.block_until_ready(c2s_bad(xarr, cx.cart3d, cx.sph3d, usys=usys))
%timeit jax.block_until_ready(c2s_bad(xarr, cx.cart3d, cx.sph3d, usys=usys))

c2s_bad(xarr, cx.cart3d, cx.sph3d, usys=usys)
```

```{note}
The `vmap` we are using is from `jaxmore`, which is a thin wrapper around `jax.vmap` that adds some extra features, in particular support for keyword arguments.
```

As expected, this is **much** slower than the hard-coded version. This is all the **wrapper overhead** and **pytree complexity** in action. Even though the argument `xarr` is a raw JAX array the vmap-with-kwargs has kwarg-related overhead and the JIT has to deal with the static arguments (`chart_from`,`chart_to`,`usys`).

### Dict[str, Array]

#### Basic JAX

```{code-cell} ipython3
def c2s_dict(x: dict[str, Array], /) -> dict[str, Array]:
    r = jnp.sqrt(x["x"] **2 + x["y"]** 2 + x["z"] ** 2)
    theta = jnp.acos(x["z"] / r)
    phi = jnp.arctan2(x["y"], x["x"])
    return {"r": r, "theta": theta, "phi": phi}

vec_c2s_dict = jax.jit(vmap(c2s_dict))

xdict = {"x": jnp.array([1.0, 4.0]), "y": jnp.array([2.0, 5.0]),
     "z": jnp.array([3.0, 6.0])}

%time jax.block_until_ready(vec_c2s_dict(xdict))
%timeit jax.block_until_ready(vec_c2s_dict(xdict))

vec_c2s_dict(xdict)
```

This is around 50% slower than the raw array version, which is expected due to the overhead of using dictionaries and the way JAX handles them as pytrees. However, this is still quite fast and may be acceptable depending on the use case.

If we want to achieve the same performance as the raw array version, we can shift the pytree conversion inside the JIT boundary. This way, the overhead of converting between pytrees and arrays is also compiled away.

```{code-cell} ipython3
@jax.jit
@vmap
def c2s_dict_comp(x: Array, /) -> Array:
    d = {"x": x[..., 0], "y": x[..., 1], "z": x[..., 2]}
    r = jnp.sqrt(d["x"] **2 + d["y"]** 2 + d["z"] ** 2)
    theta = jnp.acos(d["z"] / r)
    phi = jnp.arctan2(d["y"], d["x"])
    return jnp.stack((r, theta, phi), axis=-1)

%time jax.block_until_ready(c2s_dict_comp(xarr))
%timeit jax.block_until_ready(c2s_dict_comp(xarr))
```

This is now as fast as the raw array version, because we have minimized the overhead at the JIT boundary.

#### Coordinax

We can apply the same function to `coordinax` objects.

```{code-cell} ipython3
c2s_cx(xdict)
```

To achieve the same performance as the raw array version, we can shift the pytree conversion inside the JIT boundary. This way, the overhead of converting between pytrees and arrays is also compiled away.

```{code-cell} ipython3
from jaxmore import structured
from coordinax.internal import pack_with_usys

structurer = structured(lambda x: cx.cdict(x, cx.cart3d),
                        lambda x:pack_with_usys(x, cx.sph3d.components, usys)[0])
c2s_cx_dict = jax.jit(vmap(structurer(_c2s_cx)))

%time jax.block_until_ready(c2s_cx_dict(xarr))
%timeit jax.block_until_ready(c2s_cx_dict(xarr))

c2s_cx_dict(xarr)
```

Based on this, we can see that the actual cost of `coordinax` is paid at the JIT boundary. If we can minimize the number of times we cross the JIT boundary with `coordinax` objects, we can achieve performance that is close to raw arrays. This is a key insight for working with `coordinax` in performance-critical code: **minimize pytree conversions at the boundary**.

### Dict[str, Quantity]

#### Basic JAX

```{code-cell} ipython3
@jax.jit
@quax.quaxify  # enables Quantity support!
@vmap
def c2s_qdict(x: dict[str, u.Q], /) -> dict[str, u.Q]:
    r = jnp.sqrt(x["x"] **2 + x["y"]** 2 + x["z"] ** 2)
    theta = jnp.acos(x["z"] / r)
    phi = jnp.arctan2(x["y"], x["x"])
    return {"r": r, "theta": theta, "phi": phi}

xqdict = {k: u.Q(v, "m") for k, v in xdict.items()}

%time jax.block_until_ready(c2s_qdict(xqdict))
%timeit jax.block_until_ready(c2s_qdict(xqdict))

c2s_qdict(xqdict)
```

This is around 2-3x slower than the raw array version, which is expected due to the additional overhead of handling `Quantity` objects. Let's see if we can improve this by shifting the pytree conversion inside the JIT boundary, just like we did with the dict of arrays.

```{code-cell} ipython3
@jax.jit
@vmap
def c2s_qdict_comp(x: Array, /) -> Array:
    d = {"x": x[..., 0], "y": x[..., 1], "z": x[..., 2]}
    r = jnp.sqrt(d["x"] **2 + d["y"]** 2 + d["z"] ** 2)
    theta = jnp.acos(d["z"] / r)
    phi = jnp.arctan2(d["y"], d["x"])
    return jnp.stack((r, theta, phi), axis=-1)

%time jax.block_until_ready(c2s_qdict_comp(xarr))
%timeit jax.block_until_ready(c2s_qdict_comp(xarr))
```

The speed is now much closer to the raw array version, but now we can see the problem that was implicit in the previous transformation -- you have to manually manage units. There's a good way around this. If we stick to a particular unit system (e.g. SI), we can close over the static unit system inside the JIT-compiled function, so that you don't have to manage units at all:

```{code-cell} ipython3
import quaxed.numpy as qnp

usys = u.unitsystems.si

@jax.jit
@jax.vmap
def c2s_qdict_comp2(x: Array, /) -> Array:
    d = cx.cdict(x, usys["length"], cx.cart3d)
    r = qnp.sqrt(d["x"] **2 + d["y"]** 2 + d["z"] ** 2)
    theta = qnp.acos(d["z"] / r)
    phi = qnp.arctan2(d["y"], d["x"])
    return jnp.stack([u.ustrip(usys, r), u.ustrip(usys, theta), u.ustrip(usys, phi)], axis=-1)

%time jax.block_until_ready(c2s_qdict_comp2(xarr))
%timeit jax.block_until_ready(c2s_qdict_comp2(xarr))
```

This is now very close to the raw array version, and the user doesn't have to manage units at all, so long as they stick to the predefined unit system. On the other hand, this function is not a bit of a mess! Let's do better...

#### Coordinax

With `coordinax` all we need is:

```{code-cell} ipython3
c2s_cx(xqdict)
```

Just like the basic JAX version, this is around 2-3x slower than the raw array version due to the overhead of handling `Quantity` objects. However, as with the dict of arrays, we can shift the pytree conversion inside the JIT boundary to minimize this overhead.

```{code-cell} ipython3
# Pre-specialize `c2s_cx`, pushing pytrees into a JIT-optimizable closure.
structurer = structured(lambda x: cx.cdict(x, usys["length"], cx.cart3d),
                        lambda x: pack_with_usys(x, cx.sph3d.components, usys)[0])
c2s_cx_qdict = jax.jit(jax.vmap(structurer(_c2s_cx)))

%time jax.block_until_ready(c2s_cx_qdict(xarr))
%timeit jax.block_until_ready(c2s_cx_qdict(xarr))

c2s_cx_qdict(xarr)
```

There's no overhead, and we got to use the same function to transform `coordinax` objects with `Quantity` data, without having to manually manage units at all. This is the power of `coordinax`.

### Point

#### Basic JAX

```{code-cell} ipython3
@jax.jit
@vmap
def c2s_vec(x: cx.Point, /) -> cx.Point:
    r = quax.quaxify(jnp.sqrt)(x["x"] **2 + x["y"]** 2 + x["z"] ** 2)
    theta = quax.quaxify(jnp.acos)(x["z"] / r)
    phi = quax.quaxify(jnp.arctan2)(x["y"], x["x"])
    return replace(x, data={"r": r, "theta": theta, "phi": phi}, chart=cx.sph3d)

xvec = cx.Point.from_(xqdict, cx.cart3d)

%time jax.block_until_ready(c2s_vec(xvec))
%timeit jax.block_until_ready(c2s_vec(xvec))

c2s_vec(xvec)
```

This works, but again it's not particularly optimized. Manually optimizing is similar to the cases above.

#### Coordinax

With `coordinax` all we need is:

```{code-cell} ipython3
c2s_cx(xvec)
```

With `coordinax`, optimizing is very easy.

```{code-cell} ipython3
# Pre-specialize `c2s_cx`, pushing pytrees into a JIT-optimizable closure.
structurer = structured(lambda x: cx.Point.from_(u.Q(x, usys["length"]), cx.cart3d),
                        lambda x: pack_with_usys(x.data, cx.sph3d.components, usys)[0])
c2s_cx_vec = jax.jit(vmap(structurer(_c2s_cx)))

%time jax.block_until_ready(c2s_cx_vec(xarr))
%timeit jax.block_until_ready(c2s_cx_vec(xarr))

c2s_cx_vec(xarr)
```

</br>

---

## Jacobian of Point Maps

`jax.jacfwd` computes the forward-mode Jacobian $J^j{}_i = \partial \phi^j / \partial q^i$ of the chart transition map. We compare three approaches evaluated at a single 3-D base point.

```{code-cell} ipython3
import coordinax.charts as cxc

at = jnp.array([1.0, 0.0, 0.0])
```

### Raw JAX (baseline)

`c2s_arr` defined earlier is vmapped, so for differentiation we write the scalar version directly:

```{code-cell} ipython3
def _c2s_arr_scalar(x: Array, /) -> Array:
    r = jnp.linalg.norm(x)
    theta = jnp.acos(x[2] / r)
    phi = jnp.arctan2(x[1], x[0])
    return jnp.stack([r, theta, phi])

jac_c2s_arr = jax.jit(jax.jacfwd(_c2s_arr_scalar))

%time jac_c2s_arr(at).block_until_ready()
%timeit jac_c2s_arr(at).block_until_ready()
jac_c2s_arr(at)
```

### `pt_map` with `jax.jacfwd`

`_c2s_cx` (defined above as `cx.pt_map(cx.cart3d, cx.sph3d, usys=usys)`) is already a scalar `Array -> Array` callable that closes over the chart pair and unit system. Passing it to `jax.jacfwd` gives the same Jacobian with no extra per-call cost:

```{code-cell} ipython3
jac_pt_map_fn = jax.jit(jax.jacfwd(_c2s_cx))

%time jac_pt_map_fn(at).block_until_ready()
%timeit jac_pt_map_fn(at).block_until_ready()
jac_pt_map_fn(at)
```

There is a small one-time compilation overhead from the coordinax dispatch layer, but the per-call runtime is identical to the raw-JAX baseline.

### `jacobian_pt_map` (idiomatic API)

`cxc.jacobian_pt_map(from_chart, to_chart, usys=usys)` is the high-level API. The curried form returns a callable that wraps the full `jacfwd` pipeline, including unit handling. Wrapping it in `jax.jit` gives the same performance:

```{code-cell} ipython3
jac_fn = cxc.jacobian_pt_map(cxc.cart3d, cxc.sph3d, usys=usys)
jac_jit = jax.jit(jac_fn)

%time jac_jit(at).block_until_ready()
%timeit jac_jit(at).block_until_ready()
jac_jit(at)
```

Same runtime as the baseline. The idiomatic form also accepts quantity-valued dicts directly, without any manual unit management.
