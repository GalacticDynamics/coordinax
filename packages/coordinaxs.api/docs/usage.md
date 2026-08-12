# Usage Guide

This guide shows how to use {mod}`coordinaxs.api` to make your own types work with the `coordinax` ecosystem, without depending on `coordinax` itself.

The {mod}`coordinaxs.api` package uses [plum-dispatch](https://github.com/beartype/plum) for multiple dispatch. Each function here is declared _abstract_: it carries no implementation, only the contract. `coordinax` registers the concrete methods, and so can you.

## Registering a method

Every {mod}`coordinaxs.api` function is extended the same way: `@plum.dispatch` a function of the **same name**, annotated with the types you handle. Match the signature of the abstract function you are extending — a method whose signature does not match a real call site is never selected, and dispatch fails with `NotImplementedError` rather than a `TypeError` pointing at your code.

Here a custom container is taught to expose itself as a component dictionary by registering {func}`~coordinaxs.api.charts.cdict`, whose contract is `cdict(obj, /) -> dict`:

```python
import dataclasses

import plum
import unxt as u

import coordinaxs.api.charts as cxcapi


@dataclasses.dataclass(frozen=True)
class Station:
    """A survey station, recorded as an easting/northing offset."""

    east: u.Quantity
    north: u.Quantity


@plum.dispatch
def cdict(obj: Station, /) -> dict:
    """Expose a `Station` as 2D Cartesian components."""
    return {"x": obj.east, "y": obj.north}


station = Station(east=u.Quantity(3.0, "km"), north=u.Quantity(4.0, "km"))
print(cxcapi.cdict(station))
# {'x': Q(3., 'km'), 'y': Q(4., 'km')}
```

Nothing above imports `coordinax`. But once `coordinax` _is_ installed, the registration is live in it too — the same `plum.Function` object backs both names, so `Station` now flows into charts, points and transformations:

```python
import coordinax as cx
import coordinax.charts as cxc
import coordinax.vectors as cxv

cxv.Point.from_(cx.cdict(station), cxc.cart2d)
# Point({'x': Q(3., 'km'), 'y': Q(4., 'km')}, chart=Cart2D(M=Rn(2)))

cxc.pt_map(cx.cdict(station), cxc.cart2d, cxc.polar2d)
# {'r': Q(5., 'km'), 'theta': Q(0.9272952, 'rad')}
```

## One dispatch table per name

`plum`'s global dispatcher keys its namespace on the **bare function name**, so every module-level `@plum.dispatch def cdict` in every installed package shares one table. Two consequences:

- Registering a method is a global act. Dispatch only stays unambiguous because your method is annotated with _your_ types; never register a method whose annotations are all broad built-ins.
- Declaring the same name twice does not create a second function. It returns the same `plum.Function`, and the second declaration's docstring is dead text.

## Next Steps

- The {doc}`API Reference </packages/coordinaxs.api/api>` lists every abstract function and its contract.
- The [coordinax documentation](https://coordinax.readthedocs.io/) covers the concrete implementations, and its conventions page has more on dispatch.
