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

Generate random `coordinax.Distance` objects.

**Parameters:**

- `check_negative`: Whether to enforce non-negative distances (default: True)
- `**kwargs`: Additional arguments passed to `unxt_hypothesis.quantities`

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

## Documentation

For detailed usage examples and API documentation, see the
[full documentation](https://coordinax.readthedocs.io/).

## Contributing

Contributions are welcome! Please see the
[coordinax contributing guide](https://github.com/GalacticDynamics/coordinax/blob/main/CONTRIBUTING.md).

## License

MIT License. See [LICENSE](../../LICENSE) for details.
