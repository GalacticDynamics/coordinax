# Usage Guide

This guide demonstrates how to use `coordinax-api` to implement custom vector
types and coordinate transformations that integrate with the coordinax
ecosystem.

The `coordinax-api` package uses
[plum-dispatch](https://github.com/beartype/plum) for multiple dispatch. This
allows different implementations of the same function based on the types of the
arguments.

## Implementing Custom Vector Types

### Simple Vector Type

Here's how to create a custom 2D vector type and implement coordinate
transformations:

```
import math
from dataclasses import dataclass

import plum
from coordinax_api import vconvert


@dataclass
class MyCartesian:
    """A simple 2D vector in Cartesian coordinates."""

    x: float
    y: float


@dataclass
class MyPolar:
    """A simple 2D vector in polar coordinates."""

    r: float
    theta: float


# Implement conversion from Cartesian to Polar
@plum.dispatchpatch
def vconvert(target: MyPolar, vec: MyCartesian, **kwargs):
    """Convert Cartesian to polar coordinates."""
    r = math.sqrt(vec.x**2 + vec.y**2)
    theta = math.atan2(vec.y, vec.x)
    return MyPolar(r=r, theta=theta)


# Implement conversion from Polar to Cartesian
@plum.dispatchpatch
def vconvert(target: MyCartesian, vec: MyPolar, **kwargs):
    """Convert polar to Cartesian coordinates."""
    x = vec.r * math.cos(vec.theta)
    y = vec.r * math.sin(vec.theta)
    return MyCartesian(x=x, y=y)


# Usage
cart = MyCartesian(x=3.0, y=4.0)
polar = vconvert(MyPolar(), cart)
print(f"Polar: r={polar.r:.2f}, theta={polar.theta:.2f}")  # r=5.00, theta=0.93

back = vconvert(MyCartesian(), polar)
print(f"Cartesian: x={back.x:.2f}, y={back.y:.2f}")  # x=3.00, y=4.00
```

## Best Practices

1. **Type hints**: Always use proper type hints for dispatch to work correctly
2. **Documentation**: Document what your dispatch expects and returns
3. **Error handling**: Raise clear errors for unsupported conversions
4. **Testing**: Test your dispatches with various input types

## Next Steps

- See the [API Reference](api.md) for complete documentation
- Check out the [coordinax documentation](https://coordinax.readthedocs.io/) for
  concrete implementations
- Review the plum-dispatch documentation for advanced dispatch patterns
