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

### GalileanOp

Translates position vectors by a fixed offset:

```{code-block} text
>>> import coordinax as cx
>>> import coordinax.ops as cxo
```

```{code-block} text
>>> q = cx.Vector.from_([1, 2, 3], "kpc")
>>> op = cxo.GalileanOp.from_([10, 10, 10], "kpc")
>>> op(q)
Cart3D(x=Q(11, 'kpc'), y=Q(12, 'kpc'), z=Q(13, 'kpc'))
```

### Galilean Boost

Applies a velocity boost to a velocity vector:

```{code-block} text
>>> boost = cxo.Add.from_([1, 1, 1], "km/s")
>>> boost(u.Q(1.0, "s"), q)[1]
Cart3D(x=Q(1., 'kpc'), y=Q(2., 'kpc'), z=Q(3., 'kpc'))
```

### Rotate

Rotates vectors in space:

```{code-block} text
>>> rot = cxo.Rotate.from_euler("z", u.Q(90, "deg"))
>>> rot(q).round(2)
Cart3D(x=Q(-2., 'kpc'), y=Q(1., 'kpc'), z=Q(3., 'kpc'))
```

---

## Operator Composition and Pipes

Operators can be composed using the {class}`~coordinax.ops.Pipe` class or the
`|` operator:

```{code-block} text
>>> op1 = cxo.GalileanOp.from_([1, 0, 0], "kpc")
>>> op2 = cxo.Rotate.from_euler("z", u.Q(90, "deg"))
>>> pipe = cxo.Pipe([op1, op2])
>>> pipe(q).round(2)
Cart3D(x=Q(-2., 'kpc'), y=Q(2., 'kpc'), z=Q(3., 'kpc'))
```

Or using the pipe operator:

```{code-block} text
>>> combined = op1 | op2
>>> combined(q).round(2)
Cart3D(x=Q(-2., 'kpc'), y=Q(2., 'kpc'), z=Q(3., 'kpc'))
```

---

## Identity and Composite Operators

- {class}`~coordinax.ops.Identity`: The do-nothing operator, useful for generic
  code.
- {class}`~coordinax.ops.Pipe`: Base for building custom operator pipelines.

---

## Utilities and Advanced Usage

- {class}`~coordinax.ops.simplify`: Simplifies composed operators when possible.

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
