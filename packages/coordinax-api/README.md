# coordinax-api

Abstract dispatch API definitions for coordinax.

This package defines the abstract dispatch interfaces for coordinax's core
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

## Why Use coordinax-api?

- **Minimal dependencies**: Only depends on `plum-dispatch`
- **Extension friendly**: Easy for third-party packages to extend coordinax
- **Avoid circular dependencies**: Implement coordinax protocols without the
  full package

## When to Use

Use `coordinax-api` when you:

- Want to implement coordinax protocols in your own package
- Need coordinax integration without the full coordinax dependency
- Are building a library that provides vector types compatible with coordinax

Use the full `coordinax` package when you need concrete vector implementations
or are writing application code.

## Documentation

For detailed usage examples and API documentation, see the
[full documentation](https://coordinax.readthedocs.io/).

## License

MIT License. See [LICENSE](../../LICENSE) for details.
