# coordinaxs.api

```{toctree}
:maxdepth: 1
:hidden:

api.md
usage.md
```

Abstract dispatch API definitions for [coordinax](https://github.com/GalacticDynamics/coordinax).

This package provides the abstract dispatch interfaces for `coordinax`'s core functionality, enabling third-party packages to implement `coordinax` protocols without depending on the full `coordinax` package. This is particularly useful for library authors who want to provide `coordinax` integration without introducing a heavy dependency.

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinaxs.api
```

:::

:::{tab-item} uv

```bash
uv add coordinaxs.api
```

:::

::::

## Overview

The {mod}`coordinaxs.api` package defines abstract dispatch functions that form the public API contract for `coordinax`. By depending only on this lightweight package, you can:

- Implement custom types that work with `coordinax`
- Extend `coordinax` functionality in your own packages
- Avoid circular dependencies when integrating with `coordinax`

See the {doc}`Usage Guide </packages/coordinaxs.api/usage>` for detailed examples and the {doc}`API Reference </packages/coordinaxs.api/api>` for the list of all public API.

## Design Philosophy

The {mod}`coordinaxs.api` package follows these principles:

1. **Minimal dependencies**: Only depends on `plum` for the dispatch mechanism
2. **Abstract interfaces**: Defines what can be done, not how to do it
3. **Extension friendly**: Easy for third-party packages to extend
4. **Type-safe**: Uses Python's type system for dispatch

## When to Use

Use {mod}`coordinaxs.api` when you:

- Want to implement `coordinax` protocols in your own package
- Need `coordinax` integration without the full `coordinax` dependency
- Are building a library that provides vector types compatible with `coordinax`

Use the full `coordinax` package when you:

- Need concrete vector implementations (Cart3D, Spherical3D, etc.)
- Want to perform actual coordinate transformations
- Are writing application code rather than library code

## Related Packages

- **coordinax**: Full coordinate transformation library with concrete implementations
- **coordinaxs.hypothesis**: Hypothesis strategies for property-based testing
- **unxt**: Units and quantities library used by `coordinax`
