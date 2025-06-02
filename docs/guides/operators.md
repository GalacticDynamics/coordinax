# Operators

The {mod}`coordinax.ops` submodule provides a powerful and extensible framework
for working with vector operators. Operators are objects that represent
transformations or actions on vectors and spaces, such as translations,
rotations, and boosts. This system is inspired by mathematical operator algebra
and is designed for composability, batch operations, and JAX compatibility.

<!-- invisible-code-block: python

import unxt as u
import jax.numpy as jnp

-->

---

## Galilean Operators

Galilean operators represent the basic transformations in classical mechanics:
translations, rotations, and boosts.

### GalileanSpatialTranslation

Translates position vectors by a fixed offset:

```{code-block} python
>>> import coordinax.vecs as cxv
>>> import coordinax.ops as cxo
```

```{code-block} python
>>> q = cxv.CartesianPos3D.from_([1, 2, 3], "kpc")
>>> op = cxo.GalileanSpatialTranslation.from_([10, 10, 10], "kpc")
>>> op(q)
CartesianPos3D(
    x=Quantity(11, unit='kpc'), y=Quantity(12, unit='kpc'), z=Quantity(13, unit='kpc')
)
```

### GalileanBoost

Applies a velocity boost to a velocity vector:

```{code-block} python
>>> boost = cxo.GalileanBoost.from_([1, 1, 1], "km/s")
>>> boost(u.Quantity(1.0, "s"), q)[1]
CartesianPos3D(
    x=Quantity(1., unit='kpc'), y=Quantity(2., unit='kpc'), z=Quantity(3., unit='kpc')
)
```

### GalileanRotation

Rotates vectors in space:

```{code-block} python
>>> rot = cxo.GalileanRotation.from_euler("z", u.Quantity(90, "deg"))
>>> rot(q).round(2)
CartesianPos3D(
    x=Quantity(-2., unit='kpc'),
    y=Quantity(1., unit='kpc'),
    z=Quantity(3., unit='kpc')
)
```

---

## Operator Composition and Pipes

Operators can be composed using the {class}`~coordinax.ops.Pipe` class or the
`|` operator:

```{code-block} python
>>> op1 = cxo.GalileanSpatialTranslation.from_([1, 0, 0], "kpc")
>>> op2 = cxo.GalileanRotation.from_euler("z", u.Quantity(90, "deg"))
>>> pipe = cxo.Pipe([op1, op2])
>>> pipe(q).round(2)
CartesianPos3D(
    x=Quantity(-2., unit='kpc'), y=Quantity(2., unit='kpc'), z=Quantity(3., unit='kpc')
)
```

Or using the pipe operator:

```{code-block} python
>>> combined = op1 | op2
>>> combined(q).round(2)
CartesianPos3D(
    x=Quantity(-2., unit='kpc'), y=Quantity(2., unit='kpc'), z=Quantity(3., unit='kpc')
)
```

---

## Identity and Composite Operators

- {class}`~coordinax.ops.Identity`: The do-nothing operator, useful for generic
  code.
- {class}`~coordinax.ops.AbstractCompositeOperator`: Base for building custom
  operator pipelines.

---

## Utilities and Advanced Usage

- {class}`~coordinax.ops.simplify_op`: Simplifies composed operators when
  possible.
- {class}`~coordinax.ops.convert_to_pipe_operators`: Utility to convert a list
  of operators into a {class}`~coordinax.ops.Pipe`.

---

## Operator API and Custom Operators

All operators inherit from {class}`~coordinax.ops.AbstractOperator` and support:

- Functional call syntax: `op(vector)`
- Batch and broadcasting
- JAX transformations (jit, vmap, etc.)
- Composability

You can define your own operators by subclassing
{class}`~coordinax.ops.AbstractOperator`.

---

:::{seealso}

[API Documentation for Operators](../api/ops.md)

:::
