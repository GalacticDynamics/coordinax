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

from jaxtyping import Array
import jax
import jax.numpy as jnp
import quax

from jaxmore import vmap
import unxt as u

import coordinax.main as cx
```
