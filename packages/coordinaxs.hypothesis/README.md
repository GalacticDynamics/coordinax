# coordinaxs.hypothesis

Hypothesis strategies for property-based testing with [coordinax](https://github.com/GalacticDynamics/coordinax).

## Installation

::::{tab-set}

:::{tab-item} pip

```bash
pip install coordinaxs.hypothesis
```

:::

:::{tab-item} uv

```bash
uv add coordinaxs.hypothesis
```

:::

::::

## Quick Start

```python
from hypothesis import given
import coordinax as cx
import coordinaxs.hypothesis.main as cxst


@given(angle=cxst.angles())
def test_angle_property(angle):
    """Test that all angles are valid Angle objects."""
    assert isinstance(angle, cx.Angle)


@given(dist=cxst.distances())
def test_distance_property(dist):
    """Test that all distances are valid Distance objects."""
    assert isinstance(dist, cx.Distance)
    assert dist.value >= 0


# Calling a @given test runs it
test_angle_property()
test_distance_property()
```

## Hypothesis `st.from_type()` Support

The `coordinaxs.hypothesis` package automatically registers strategies for distance types with Hypothesis's `st.from_type()` function, allowing them to be used seamlessly in property-based tests without explicitly importing the strategies.

### Distance Types

The following distance types are registered:

- `coordinax.distances.Distance` → uses `distances()` strategy
- `coordinaxs.astro.DistanceModulus` → uses `distance_moduli()` strategy (registered on importing `coordinaxs.hypothesis.astro`)
- `coordinaxs.astro.Parallax` → uses `parallaxes()` strategy (registered on importing `coordinaxs.hypothesis.astro`)

**Examples:**

```python
from hypothesis import given, strategies as st
import coordinax.distances as cxd
import coordinaxs.astro as cxastro
import coordinaxs.hypothesis.astro  # registers the astro types with from_type


# Distance types automatically work with from_type
@given(dist=st.from_type(cxd.Distance))
def test_distance(dist):
    assert isinstance(dist, cxd.Distance)
    assert dist.value >= 0  # default check_negative=True


@given(dm=st.from_type(cxastro.DistanceModulus))
def test_distance_modulus(dm):
    assert isinstance(dm, cxastro.DistanceModulus)
    assert dm.unit == "mag"


@given(plx=st.from_type(cxastro.Parallax))
def test_parallax(plx):
    assert isinstance(plx, cxastro.Parallax)
    assert plx.value >= 0  # default check_negative=True


# from_type works seamlessly with st.builds()
def compute_distance_from_parallax(plx: cxastro.Parallax) -> cxd.Distance:
    """Convert parallax to distance."""
    return cxd.Distance.from_(1 / plx.value, "pc")


@given(
    result=st.builds(compute_distance_from_parallax, plx=st.from_type(cxastro.Parallax))
)
def test_parallax_to_distance(result):
    assert isinstance(result, cxd.Distance)


test_distance()
test_distance_modulus()
test_parallax()
test_parallax_to_distance()
```

## Documentation

For detailed usage examples and API documentation, see the [full documentation](https://coordinax.readthedocs.io/).

## Contributing

Contributions are welcome! Please see the [coordinax contributing guide](https://github.com/GalacticDynamics/coordinax/blob/main/CONTRIBUTING.md).

## License

MIT License. See [LICENSE](../../LICENSE) for details.
