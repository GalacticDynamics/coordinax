# coordinax-hypothesis

Hypothesis strategies for property-based testing with
[coordinax](https://github.com/GalacticDynamics/coordinax).

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinax-hypothesis
```

:::

:::{tab-item} uv

```bash
uv add coordinax-hypothesis
```

:::

::::

## Quick Start

```python
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


@given(angle=cxst.angles())
def test_angle_property(angle):
    """Test that all angles are valid Angle objects."""
    assert isinstance(angle, cx.Angle)


@given(dist=cxst.distances())
def test_distance_property(dist):
    """Test that all distances are valid Distance objects."""
    assert isinstance(dist, cx.Distance)
    assert dist.value >= 0
```

## Available Strategies

### `representation_classes(filter=object, exclude_abstract=True)`

Generate random representation class types from `coordinax`.

**Parameters:**

- `filter`: A class or tuple of classes to limit the representations to, or a
  strategy generating such values. Use dimensional flags like
  `cx.r.Abstract1D`, `cx.r.Abstract2D`, `cx.r.Abstract3D`, or more specific
  mixins like `cx.r.AbstractSpherical3D`. Tuples apply all filters
  simultaneously (default: `object` includes all)
- `exclude_abstract`: Whether to exclude abstract classes (default: `True`)

**Examples:**

```python
from hypothesis import given, strategies as st
import coordinax as cx
import coordinax_hypothesis as cxst


# Any representation class
@given(rep_class=cxst.representation_classes())
def test_any_rep(rep_class):
    assert issubclass(rep_class, cx.r.AbstractRep)


# Only 3D representations
@given(rep_class=cxst.representation_classes(filter=cx.r.Abstract3D))
def test_3d_rep(rep_class):
    assert issubclass(rep_class, cx.r.Abstract3D)


# Only spherical 3D representations
@given(
    rep_class=cxst.representation_classes(
        filter=(cx.r.Abstract3D, cx.r.AbstractSpherical3D)
    )
)
def test_spherical_3d_rep(rep_class):
    assert issubclass(rep_class, (cx.r.Abstract3D, cx.r.AbstractSpherical3D))


# Dynamically choose dimensionality
@given(
    rep_class=cxst.representation_classes(
        filter=st.sampled_from(
            [
                cx.r.Abstract1D,
                cx.r.Abstract2D,
                cx.r.Abstract3D,
            ]
        )
    )
)
def test_dynamic(rep_class):
    assert issubclass(rep_class, (cx.r.Abstract1D, cx.r.Abstract2D, cx.r.Abstract3D))
```

### `angles(wrap_to=None, **kwargs)`

Generate random `unxt.Angle` objects.

**Parameters:**

- `wrap_to`: Optional strategy for generating wrapping bounds
- `**kwargs`: Additional arguments passed to `unxt_hypothesis.quantities`

**Examples:**

```python
from hypothesis import given, strategies as st
import unxt as u
import coordinax_hypothesis as cxst


# Basic angles
@given(angle=cxst.angles())
def test_basic(angle):
    assert isinstance(angle, u.Angle)


# Angles in degrees
@given(angle=cxst.angles(units="deg"))
def test_degrees(angle):
    assert angle.unit == "deg"


# Angles with wrapping
@given(angle=cxst.angles(wrap_to=st.just((u.Q(0, "deg"), u.Q(360, "deg")))))
def test_wrapped(angle):
    assert angle.wrap_to is not None
```

### `distances(check_negative=True, **kwargs)`

Generate random {class}`coordinax.Distance` objects.

**Parameters:**

- `check_negative`: Whether to enforce non-negative distances (default: True)
- `**kwargs`: Additional arguments passed to {func}`unxt_hypothesis.quantities`

**Examples:**

```python
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


# Basic non-negative distances
@given(dist=cxst.distances())
def test_basic(dist):
    assert isinstance(dist, cx.Distance)
    assert dist.value >= 0


# Distances in specific units
@given(dist=cxst.distances(units="kpc"))
def test_kpc(dist):
    assert dist.unit == "kpc"


# Allow negative distances
@given(dist=cxst.distances(check_negative=False))
def test_signed(dist):
    assert isinstance(dist, cx.Distance)
```

### `distance_moduli(**kwargs)`

Generate random `coordinax.DistanceModulus` objects.

**Parameters:**

- `**kwargs`: Additional arguments passed to `unxt_hypothesis.quantities`

**Examples:**

```python
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


# Basic distance moduli
@given(dm=cxst.distance_moduli())
def test_basic(dm):
    assert isinstance(dm, cx.DistanceModulus)
    assert dm.unit == "mag"


# Distance modulus arrays
@given(dm=cxst.distance_moduli(shape=10))
def test_array(dm):
    assert dm.shape == (10,)
```

### `parallaxes(check_negative=True, **kwargs)`

Generate random `coordinax.Parallax` objects.

**Parameters:**

- `check_negative`: Whether to enforce non-negative parallaxes (default: True)
- `**kwargs`: Additional arguments passed to `unxt_hypothesis.quantities`

**Examples:**

```python
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


# Basic non-negative parallaxes
@given(plx=cxst.parallaxes())
def test_basic(plx):
    assert isinstance(plx, cx.Parallax)
    assert plx.value >= 0


# Parallaxes in milliarcseconds
@given(plx=cxst.parallaxes(units="mas"))
def test_mas(plx):
    assert plx.unit == "mas"


# Allow negative parallaxes (noisy measurements)
@given(plx=cxst.parallaxes(check_negative=False))
def test_noisy(plx):
    assert isinstance(plx, cx.Parallax)
```

### `representations(filter=(), dimensionality=None)`

Generate random representation instances from `coordinax`.

**Parameters:**

- `filter`: A class or tuple of classes to limit the representations to, or a
  strategy generating such values (default: `()` applies no extra filter). For
  example, `cx.r.Abstract1D` for 1D representations, `cx.r.Abstract2D` for 2D,
  or `cx.r.Abstract3D` for 3D. Tuples apply all filters simultaneously.
- `dimensionality`: Constraint on representation dimensionality. Can be:
  - `None`: No constraint
  - An integer: Exact dimensionality match (e.g., `dimensionality=2`)
  - A strategy: Draw dimensionality from strategy (e.g.,
    `st.integers(min_value=1, max_value=2)`)
  - Default: `None`

**Examples:**

```python
from hypothesis import given, strategies as st
import coordinax as cx
import coordinax_hypothesis as cxst


# Any representation instance
@given(rep=cxst.representations())
def test_any_rep(rep):
    assert isinstance(rep, cx.r.AbstractRep)

# Only 3D representations
@given(rep=cxst.representations(filter=cx.r.Abstract3D))
def test_3d_rep(rep):
    assert isinstance(rep, cx.r.Abstract3D)


# Spherical 3D representations
@given(rep=cxst.representations(filter=(cx.r.Abstract3D, cx.r.AbstractSpherical3D)))
def test_spherical_3d(rep):
    assert isinstance(rep, cx.r.Abstract3D)
    assert isinstance(rep, cx.r.AbstractSpherical3D)


# Exact dimensionality
@given(rep=cxst.representations(dimensionality=2))
def test_exact_2d(rep):
    assert rep.dimensionality == 2


# Dimensionality using strategy
@given(rep=cxst.representations(dimensionality=st.integers(min_value=1, max_value=2)))
def test_strategy_dim(rep):
    assert 1 <= rep.dimensionality <= 2


# Include 0-dimensional representations
@given(rep=cxst.representations(dimensionality=None, exclude=()))
def test_with_0d(rep):
    assert isinstance(rep, cx.r.AbstractRep)


# Dynamically choose dimensionality
@given(
    rep=cxst.representations(filter=st.sampled_from([cx.r.Abstract1D, cx.r.Abstract2D]))
)
def test_dynamic_dim(rep):
    assert isinstance(rep, (cx.r.Abstract1D, cx.r.Abstract2D))
```

### `representations_like(representation)`

Generate representations matching the flags of a template representation.

This strategy inspects a template representation to determine its type flags
(e.g., `Abstract1D`, `Abstract2D`, `Abstract3D`, `AbstractSpherical3D`, etc.) and
dimensionality, then generates new representations matching those same criteria.
This is useful for generating varied test cases while preserving key structural
properties.

**Parameters:**

- `representation`: A representation instance to use as a template, or a
  strategy that generates one. The generated representations will match all the
  flags and dimensionality of the template.

**Examples:**

```python
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


# Generate 3D representations like Cart3D
@given(rep=cxst.representations_like(cx.r.cart3d))
def test_3d_rep(rep):
    assert isinstance(rep, cx.r.Abstract3D)
    assert rep.dimensionality == 3
    # Could be Cart3D, Spherical3D, Cylindrical3D, etc.


# Generate 2D representations like Polar2D
@given(rep=cxst.representations_like(cx.r.polar2d))
def test_2d_rep(rep):
    assert isinstance(rep, cx.r.Abstract2D)
    assert rep.dimensionality == 2
    # Could be Cart2D, Polar2D, TwoSphere, etc.


# Generate 1D representations
@given(rep=cxst.representations_like(cx.r.radial1d))
def test_1d_rep(rep):
    assert isinstance(rep, cx.r.Abstract1D)
    assert rep.dimensionality == 1


# Use with a dynamic template
@given(
    template=cxst.representations(filter=cx.r.Abstract3D),
    rep=cxst.representations_like(cxst.representations(filter=cx.r.Abstract3D)),
)
def test_matching_3d(template, rep):
    assert rep.dimensionality == template.dimensionality
    assert isinstance(rep, cx.r.Abstract3D)
```

### `representation_time_chain(role, rep)`

Generate a chain of representations following the time antiderivative pattern.

Given a role flag (position, velocity, or acceleration) and a representation,
this strategy returns a tuple containing representations that match the flags
of each time antiderivative up to and including a position representation. Each
element in the chain is generated using `representations_like()` to match the
flags of the corresponding time antiderivative.

This is particularly useful for testing coordinate transformations across
different time derivatives (e.g., converting from acceleration to velocity to
position).

**Parameters:**

- `role`: The starting role flag (`cx.r.Pos`, `cx.r.Vel`, or `cx.r.Acc`).
- `rep`: The starting representation or a strategy that generates one.

**Returns:**

- A tuple of representations following the time antiderivative chain:
  - If input is position: `(pos_rep,)`
  - If input is velocity: `(vel_rep, pos_rep)`
  - If input is acceleration: `(acc_rep, vel_rep, pos_rep)`

**Examples:**

```python
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


# Generate a chain from acceleration
@given(chain=cxst.representation_time_chain(cx.r.Acc, cx.r.cart3d))
def test_acc_chain(chain):
    acc_rep, vel_rep, pos_rep = chain
    # All are 3D Cartesian-like representations
    assert isinstance(acc_rep, cx.r.Abstract3D)
    assert isinstance(vel_rep, cx.r.Abstract3D)
    assert isinstance(pos_rep, cx.r.Abstract3D)


# Generate a chain from velocity
@given(chain=cxst.representation_time_chain(cx.r.Vel, cx.r.polar2d))
def test_vel_chain(chain):
    vel_rep, pos_rep = chain
    # All are 2D representations
    assert isinstance(vel_rep, cx.r.Abstract2D)
    assert isinstance(pos_rep, cx.r.Abstract2D)


# Position just returns itself
@given(chain=cxst.representation_time_chain(cx.r.Pos, cx.r.sph3d))
def test_pos_chain(chain):
    (pos_rep,) = chain
    assert isinstance(pos_rep, cx.r.Abstract3D)
```

### `vectors_with_target_rep(rep=representations(), role=cx.r.Pos, dtype=jnp.float32, shape=(), elements=None)`

Generate a vector and a time-derivative chain with matching flags.

This strategy is useful for testing conversion operations where you need a
source vector and a full set of target representations (following the time
antiderivative chain) that it can be converted to. The target chain
automatically matches the flags of the source vector's representation.

**Parameters:**

- `rep`: A representation instance or strategy for the source vector (default:
  uses `representations()` strategy)
- `role`: The role flag for the source vector (`cx.r.Pos`, `cx.r.Vel`,
  `cx.r.Acc`)
- `dtype`: The data type for array components (default: `jnp.float32`)
- `shape`: The shape for the vector components (default: scalar shape `()`)
- `elements`: Strategy for generating element values

**Returns:**

- A tuple of `(vector, target_chain)` where `target_chain` is a tuple of
  representations following the time antiderivative pattern, all matching the
  flags of the source vector's representation.

**Examples:**

```python
from hypothesis import given
import coordinax as cx
import coordinax_hypothesis as cxst


# Test vector conversions to a full chain of targets
@given(vec_and_chain=cxst.vectors_with_target_rep(rep=cx.r.cart3d, role=cx.r.Pos))
def test_position_conversion(vec_and_chain):
    vec, target_chain = vec_and_chain
    # target_chain is just (pos_rep,) for position sources
    (target_rep,) = target_chain
    converted = vec.vconvert(target_rep)
    assert converted.rep == target_rep


# Test velocity vector with full chain (requires a position vector)
@given(
    vec_and_chain=cxst.vectors_with_target_rep(rep=cx.r.cart3d, role=cx.r.Vel),
    pos_vec=cxst.vectors(rep=cx.r.cart3d, role=cx.r.Pos),
)
def test_velocity_conversion_chain(vec_and_chain, pos_vec):
    vec, target_chain = vec_and_chain
    # target_chain is (vel_rep, pos_rep)
    for target_rep in target_chain:
        converted = vec.vconvert(target_rep, pos_vec)
        assert converted.rep == target_rep
```

## Hypothesis `st.from_type()` Support

The `coordinax-hypothesis` package automatically registers strategies for
distance types with Hypothesis's `st.from_type()` function, allowing them to be
used seamlessly in property-based tests without explicitly importing the
strategies.

### Distance Types

The following coordinax distance types are registered:

- `coordinax.Distance` → uses `distances()` strategy
- `coordinax.DistanceModulus` → uses `distance_moduli()` strategy
- `coordinax.Parallax` → uses `parallaxes()` strategy

**Examples:**

```python
from hypothesis import given, strategies as st
import coordinax.distance as cxd


# Distance types automatically work with from_type
@given(dist=st.from_type(cxd.Distance))
def test_distance(dist):
    assert isinstance(dist, cxd.Distance)
    assert dist.value >= 0  # default check_negative=True


@given(dm=st.from_type(cxd.DistanceModulus))
def test_distance_modulus(dm):
    assert isinstance(dm, cxd.DistanceModulus)
    assert dm.unit == "mag"


@given(plx=st.from_type(cxd.Parallax))
def test_parallax(plx):
    assert isinstance(plx, cxd.Parallax)
    assert plx.value >= 0  # default check_negative=True


# from_type works seamlessly with st.builds()
def compute_distance_from_parallax(plx: cxd.Parallax) -> cxd.Distance:
    """Convert parallax to distance."""
    return cxd.Distance.from_(1 / plx.value, "pc")


@given(result=st.builds(compute_distance_from_parallax, plx=st.from_type(cxd.Parallax)))
def test_parallax_to_distance(result):
    assert isinstance(result, cxd.Distance)
```

### `vectors(rep=representations(), role=cx.r.Pos, dtype=jnp.float32, shape=(), elements=None)`

Generate random `coordinax.Vector` instances.

**Parameters:**

- `rep`: A representation instance or strategy to generate one (default: uses
  `representations()` strategy). This determines the coordinate chart and
  dimensionality of the vector.
- `role`: The role flag for the vector (`cx.r.Pos`, `cx.r.Vel`, `cx.r.Acc`).
- `dtype`: The data type for array components (default: `jnp.float32`). Can be a
  dtype or a strategy.
- `shape`: The shape for the vector components. Can be an integer (for 1D), a
  tuple of integers, or a strategy (default: scalar shape `()`).
- `elements`: Strategy for generating element values. If None, uses finite
  floats.

**Note:** This strategy requires {mod}`unxt-hypothesis` to be installed for
generating Quantity components.

**Examples:**

```python
import jax.numpy as jnp
from hypothesis import given, strategies as st
import coordinax as cx
import coordinax_hypothesis as cxst


# Generate any vector
@given(vec=cxst.vectors())
def test_any_vector(vec):
    assert isinstance(vec, cx.vecs.Vector)


# Generate vectors with a specific representation
@given(vec=cxst.vectors(rep=cx.r.cart3d))
def test_cartesian_3d(vec):
    assert vec.rep == cx.r.cart3d
    assert isinstance(vec, cx.vecs.Vector)


# Generate batched vectors
@given(vec=cxst.vectors(shape=(10,)))
def test_batched_vectors(vec):
    assert vec.shape == (10,)


# Generate position vectors only
@given(vec=cxst.vectors(rep=cxst.representations(filter=cx.r.Abstract3D), role=cx.r.Pos))
def test_position_vectors(vec):
    assert isinstance(vec.role, cx.r.Pos)
    assert isinstance(vec.rep, cx.r.Abstract3D)


# Generate 3D velocity vectors
@given(
    vec=cxst.vectors(
        rep=cxst.representations(filter=cx.r.Abstract3D),
        role=cx.r.Vel,
    )
)
def test_3d_velocity_vectors(vec):
    assert isinstance(vec.role, cx.r.Vel)
    assert isinstance(vec.rep, cx.r.Abstract3D)


# Generate vectors with specific dtype and shape
@given(vec=cxst.vectors(dtype=jnp.float32, shape=(5, 3)))
def test_float32_batched(vec):
    assert vec.shape == (5, 3)
```

## Documentation

For detailed usage examples and API documentation, see the
[full documentation](https://coordinax.readthedocs.io/).

## Contributing

Contributions are welcome! Please see the
[coordinax contributing guide](https://github.com/GalacticDynamics/coordinax/blob/main/CONTRIBUTING.md).

## License

MIT License. See [LICENSE](../../LICENSE) for details.
