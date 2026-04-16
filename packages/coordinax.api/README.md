# coordinax.api

Abstract dispatch API definitions for `coordinax`.

This package defines the abstract dispatch interfaces for `coordinax`'s core functionality, enabling third-party packages to implement `coordinax` protocols without depending on the full `coordinax` package. This is particularly useful for library authors who want to provide `coordinax` integration without introducing a heavy dependency.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinax.api
```

:::

:::{tab-item} uv

```bash
uv add coordinax.api
```

:::

::::

## Why Use `coordinax.api`?

- **Minimal dependencies**: Only depends on `plum`
- **Extension friendly**: Easy for third-party packages to extend `coordinax`
- **Avoid circular dependencies**: Implement `coordinax` protocols without the full package

## When to Use

Use `coordinax.api` when you:

- Want to implement `coordinax` protocols in your own package
- Need `coordinax` integration without the full `coordinax` dependency
- Are building a library that provides vector types compatible with `coordinax`

Use the full `coordinax` package when you need concrete vector implementations or are writing application code.

## Quick Example

`coordinax.api` exposes abstract multiple-dispatch functions that you extend in your own package.

```pycon
>>> import coordinax.api.charts as cxcapi
>>> cxcapi.guess_chart
<multiple-dispatch function guess_chart ...>
```

Register a minimal implementation in your own package, then call it through the `coordinax.api` namespace:

```pycon
>>> import coordinax.api.representations as cxrapi
>>> import dataclasses
>>> import math
>>> import plum

>>> @dataclasses.dataclass
... class MyCartesian:
...     x: float
...     y: float
...

>>> @dataclasses.dataclass
... class MyPolar:
...     r: float
...     theta: float
...

>>> @plum.dispatch
... def cconvert(target: type[MyPolar], vec: MyCartesian, /, **kwargs: object) -> MyPolar:
...     return MyPolar(r=(vec.x**2 + vec.y**2) ** 0.5, theta=math.atan2(vec.y, vec.x))
...

>>> vec = MyCartesian(1.0, 2.0)
>>> cxrapi.cconvert(MyPolar, vec)
MyPolar(r=2.23606797749979, theta=1.1071487177940904)
```

## Documentation

For detailed usage examples and API documentation, see the [full documentation](https://coordinax.readthedocs.io/).

## License

MIT License. See [LICENSE](../../LICENSE) for details.
