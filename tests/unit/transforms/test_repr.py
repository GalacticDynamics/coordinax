"""A transform whose field is a quantity must still be representable."""

__all__: tuple[str, ...] = ()

from typing import Any

import unxt as u

import coordinax.transforms as cxfm


class _QuantityFieldTransform(cxfm.AbstractTransform):
    """A transform holding a bare quantity, with a quantity default.

    None of the shipped transforms has this shape, which is why the bug this
    guards reached a release: `__pdoc__` drops fields equal to their class
    default, and comparing two quantities yields a dimensionless *quantity*
    under unxt v2, not a bare array, which `jax.numpy.all` rejects.
    """

    omega: u.AbstractQuantity = u.Q(1.0, "1/Gyr")

    @classmethod
    def groups(cls) -> frozenset[type]:
        return frozenset((cxfm.groups.DiffeomorphismGroup,))

    @property
    def inverse(self) -> "_QuantityFieldTransform":
        return _QuantityFieldTransform(omega=-self.omega)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_repr_with_a_quantity_field_equal_to_its_default() -> None:
    """`jax.numpy.all` rejects a quantity; the filter must tolerate one."""
    assert "_QuantityFieldTransform" in repr(_QuantityFieldTransform())


def test_repr_with_a_quantity_field_differing_from_its_default() -> None:
    """The other branch of the same comparison."""
    op = _QuantityFieldTransform(omega=u.Q(2.0, "1/Gyr"))
    assert "_QuantityFieldTransform" in repr(op)
