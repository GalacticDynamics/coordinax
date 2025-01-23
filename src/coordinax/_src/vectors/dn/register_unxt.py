"""Built-in vector classes."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method

import unxt as u

from .cartesian import CartesianPosND


@conversion_method(CartesianPosND, u.Quantity)  # type: ignore[arg-type]
def vec_to_q(obj: CartesianPosND, /) -> Shaped[u.Quantity["length"], "*batch N"]:
    """`coordinax.AbstractPos3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], unit="km"))
    >>> convert(vec, u.Quantity)
    Quantity['length'](Array([1, 2, 3, 4, 5], dtype=int32), unit='km')

    """
    return obj.q
