# Project Overview

This is a UV workspace repository containing multiple packages:

- **coordinax**: Main library for coordinates in JAX with support for JIT
  compilation, auto-differentiation, vectorization, and GPU/TPU acceleration
- **coordinax-api**: Abstract dispatch API that defines the multiple-dispatch
  interfaces implemented by `coordinax` and other packages. Minimal dependencies
  (only `plum-dispatch`).
- **coordinax-astro**: Astronomy-specific reference frames (ICRS,
  Galactocentric, etc.) for coordinax
- **coordinax-hypothesis**: Hypothesis strategies for property-based testing
  with `coordinax`

## Main Package: coordinax

- **Language**: Python 3.11+
- **Main API**: Vector types, coordinate transformations, and reference frames
  - Vector types: `CartesianPos3D`, `SphericalPos`, `CylindricalPos`, etc.
  - Angle and Distance types with units via `unxt`
  - `vconvert()`: Transform between coordinate representations
  - Operators on vectors: `GalileanRotateOp`, `GalileanBoostOp`, etc.
  - Frame and their transformations: `frames.frame_transform_op()`
  - Coordinates with frames: `Coordinate`.
- **Design goals**: JAX-compatible coordinate objects, type-safe
  transformations, seamless integration with existing JAX code via Quax
- **JAX integration**: Objects are PyTrees via Equinox. Use `quaxed` for
  pre-quaxified JAX functions. Performant with JIT, vmap, grad.

## Architecture & Core Components

- **Vector types** (hierarchical):
  - `AbstractPos`: Base class for position vectors
  - `AbstractVel`: Base class for velocity vectors
  - `AbstractAcc`: Base class for acceleration vectors
  - Concrete implementations: Cartesian, Spherical, Cylindrical, etc.
  - All vectors are `ArrayValue` subclasses (Quax protocol) for JAX integration
- **Angle and Distance types**: Specialized scalar types with units
  - `Angle`: Angular quantities with wrapping support
  - `Distance`: Distance quantities
  - Both integrate with `unxt` for unit handling
- **Transformations**: Multiple dispatch-based coordinate conversions
  - `vconvert(target_type, vector, ...)`: Convert between representations
  - Automatic Jacobian computation for velocity transformations
- **Reference frames**: Astronomical coordinate systems (in `coordinax-astro`)
  - ICRS, Galactocentric, etc.
  - Frame transformations via `frame_transform_op()`
- **Operators**: Frame-aware operations on vectors
  - `GalileanRotateOp`, `GalileanBoostOp`, etc.

## Folder Structure

### Root Level (UV Workspace)

- `/src/coordinax/`: Main package public API with re-exports
- `/packages/`: Workspace packages
  - `coordinax-api/`: Abstract dispatch API package
  - `coordinax-astro/`: Astronomy-specific frames and transformations
  - `coordinax-hypothesis/`: Hypothesis strategies package
- `/tests/`: Main package tests, organized into `unit/`, `integration/`,
  `benchmark/`
- `README.md`: Main package documentation, tested via Sybil (all Python code
  blocks are doctests)
- `conftest.py`: Pytest config, Sybil setup, optional dependency handling
- `noxfile.py`: Task automation with dependency groups
- `pyproject.toml`: Root workspace configuration with `[tool.uv.workspace]`

### Main Package Structure (`/src/coordinax/`)

- `_src/`: Private implementation code
  - `vectors/`: Vector classes (position, velocity, acceleration)
  - `angles.py`: Angle type implementation
  - `distances/`: Distance type implementations
  - `frames/`: Reference frame definitions and transformations
  - `operators/`: Frame-aware operators
- `_coordinax_space_frames/`: Frame-specific coordinate spaces
- `_interop/`: Optional dependency integrations
- Public API files: `vecs.py`, `ops.py`, `frames.py`, `angle.py`, `distance.py`

## Coding Style

- Always use type hints (standard typing, `jaxtyping.Array`, `ArrayLike`, shape
  annotations)
- Extensive use of Plum multiple dispatch - check `.methods` on any function to
  see all dispatches
- Runtime type checking via `beartype` for validation
- Immutability is a core constraint: methods return new objects, never mutate
- Keep dependencies minimal; the core dependencies are listed in
  `pyproject.toml`
- Docstrings should be concise and include testable usage examples
- `__all__` should always be a tuple (not list) unless it needs to be mutated
  with `+=` - prefer immutable by default
- Prefer `u.Q` over `u.Quantity` for creating quantities (shorter and more
  concise)

### JAX Integration via Quax

- Vectors are `ArrayValue` subclasses (Quax protocol)
- PyTree registration handled automatically via Equinox
- Use `quaxed` library (pre-quaxified JAX) for convenience, or manually apply
  `quax.quaxify` decorator
- Mixins from `quax-blocks` provide operator overloading (`NumpyBinaryOpsMixin`,
  etc.)

### Immutability

- All vector operations return new instances
- Use `dataclassish.replace()` for attribute updates
- Follow Equinox patterns for JAX compatibility

### Import Hook

- `setup_package.py` installs jaxtyping import hook for runtime checking
- Not required for normal usage but enables beartype integration during tests

## Tooling

- This repo uses `uv` for dependency and environment management
- This repo uses `nox` for all development tasks
- Before committing, run full checks:
  ```bash
  uv run nox -s all
  ```
- Common sessions:
  - `nox -s lint`: pre-commit + pylint
  - `nox -s test`: pytest suite
  - `nox -s docs`: build documentation (add `--serve` to preview)
  - `nox -s pytest_benchmark`: run CodSpeed benchmarks

## Testing

- Use `pytest` for all test suites with Sybil for doctests in code and markdown
- Add unit tests for every new function or class
- Test organization: `unit/`, `integration/`, `benchmark/`
- Optional dependencies handled via
  `optional_dependencies.OptionalDependencyEnum`
  - Tests requiring optional deps auto-skip if not installed
  - `conftest.py` manages `collect_ignore_glob` for missing deps
- For JAX-related behavior:
  - Confirm PyTree registration works correctly (flatten/unflatten)
  - Verify compatibility with transformations like `jit`, `vmap`, and `grad`
  - Test numerical accuracy where applicable (e.g., coordinate transformations)
  - Tests should run on CPU by default; no accelerators required
- Hypothesis for property-based testing of coordinate transformation laws

## Optional Dependencies

Optional interop groups:

- `astro`: Astronomy-specific frames (installs `coordinax-astro`)

Install with: `uv add coordinax --extra astro`

## Workspace Packages

This repository uses a UV workspace structure with multiple packages (e.g.,
`coordinax`, `coordinax-api`, `coordinax-astro`, `coordinax-hypothesis`). When
creating new workspace packages, use this versioning setup pattern:

```toml
[build-system]
build-backend = "hatchling.build"
requires      = ["hatch-vcs", "hatchling"]

[tool.hatch.version]
raw-options = { root = "../..", search_parent_directories = true, git_describe_command = "git describe --dirty --tags --long --match '<package-name>-v*'", local_scheme = "no-local-version" }
source      = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/<package_name>/_version.py"
version-file-template = """\
version: str = {version!r}
version_tuple: tuple[int, int, int] | tuple[int, int, int, str, str]
version_tuple = {version_tuple!r}
"""

[tool.uv.sources]
coordinax = { workspace = true }
```

Replace `<package-name>` with the actual package name (e.g.,
`coordinax-hypothesis-v*`) and `<package_name>` with the Python module name
(e.g., `coordinax_hypothesis`). This enables automatic versioning from git tags.

## Final Notes

Preserve JAX compatibility and immutability above all. When extending coordinate
types or transformations, ensure type safety and test with JAX transformations.
Follow Equinox/Quax patterns for custom array types. Coordinate transformations
should be numerically accurate and well-tested. Documentation examples must be
executable (they're tested).
