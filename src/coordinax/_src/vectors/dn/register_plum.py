"""Built-in vector classes."""

__all__: list[str] = []


from jaxtyping import Shaped
from plum import conversion_method

from unxt.quantity import AbstractQuantity

from .cartesian import CartesianAccND, CartesianPosND, CartesianVelND


@conversion_method(CartesianAccND, AbstractQuantity)  # type: ignore[arg-type]
@conversion_method(CartesianVelND, AbstractQuantity)
@conversion_method(CartesianPosND, AbstractQuantity)
def vec_to_q(
    obj: CartesianPosND | CartesianVelND | CartesianAccND, /
) -> Shaped[AbstractQuantity, "*batch N"]:
    """`coordinax.AbstractPos3D` -> `unxt.Quantity`.

    Examples
    --------
    >>> from plum import convert
    >>> import unxt as u
    >>> import coordinax as cx

    >>> vec = cx.vecs.CartesianPosND(u.Quantity([1, 2, 3, 4, 5], unit="km"))
    >>> convert(vec, u.Quantity)
    Quantity['length'](Array([1, 2, 3, 4, 5], dtype=int32), unit='km')

    >>> vec = cx.vecs.CartesianVelND(u.Quantity([1, 2, 3, 4, 5], unit="km/s"))
    >>> convert(vec, u.Quantity)
    Quantity['speed'](Array([1, 2, 3, 4, 5], dtype=int32), unit='km / s')

    >>> vec = cx.vecs.CartesianAccND(u.Quantity([1, 2, 3, 4, 5], unit="km/s2"))
    >>> convert(vec, u.Quantity)
    Quantity['acceleration'](Array([1, 2, 3, 4, 5], dtype=int32), unit='km / s2')

    """
    return obj.q
