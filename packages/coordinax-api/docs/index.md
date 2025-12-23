# coordinax-api

```{toctree}
:maxdepth: 1
:hidden:

api
usage
```

Abstract dispatch API definitions for
[coordinax](https://github.com/GalacticDynamics/coordinax).

This package provides the abstract dispatch interfaces for coordinax's core
functionality, enabling third-party packages to implement coordinax protocols
without depending on the full coordinax package. This is particularly useful for
library authors who want to provide coordinax integration without introducing a
heavy dependency.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinax-api
```

:::

:::{tab-item} uv

```bash
uv add coordinax-api
```

:::

::::

## Overview

The `coordinax-api` package defines abstract dispatch functions that form the
public API contract for coordinax. By depending only on this lightweight
package, you can:

- Implement custom vector types that work with coordinax
- Extend coordinax functionality in your own packages
- Avoid circular dependencies when integrating with coordinax

## Quick Example

```python
from plum import dispatch
from coordinax_api import vconvert


# Define your custom vector type
class MyVector:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Implement vconvert for your type
@dispatch
def vconvert(target: type[dict], vec: MyVector, **kwargs) -> dict:
    return {"x": vec.x, "y": vec.y}


# Now you can use vconvert with your type
vec = MyVector(1.0, 2.0)
result = vconvert(dict, vec)
print(result)  # {'x': 1.0, 'y': 2.0}
```

## API Functions

### `vconvert`

The primary dispatch function for vector coordinate transformations.

```{eval-rst}
.. currentmodule:: coordinax_api

.. autofunction:: vconvert
```

See the [Usage Guide](usage.md) for more detailed examples and the
[API Reference](api.md) for complete documentation.

## Design Philosophy

The `coordinax-api` package follows these principles:

1. **Minimal dependencies**: Only depends on `plum-dispatch` for the dispatch
   mechanism
2. **Abstract interfaces**: Defines what can be done, not how to do it
3. **Extension friendly**: Easy for third-party packages to extend
4. **Type-safe**: Uses Python's type system for dispatch

## When to Use

Use `coordinax-api` when you:

- Want to implement coordinax protocols in your own package
- Need coordinax integration without the full coordinax dependency
- Are building a library that provides vector types compatible with coordinax

Use the full `coordinax` package when you:

- Need concrete vector implementations (CartesianPos3D, SphericalPos, etc.)
- Want to perform actual coordinate transformations
- Are writing application code rather than library code

## Related Packages

- **coordinax**: Full coordinate transformation library with concrete
  implementations
- **coordinax-hypothesis**: Hypothesis strategies for property-based testing
- **unxt**: Units and quantities library used by coordinax

## License

MIT License. See [LICENSE](../../LICENSE) for details.
