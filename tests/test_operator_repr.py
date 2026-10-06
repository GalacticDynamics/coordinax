"""An operator whose field is a quantity must still be representable."""

from typing import Any

import unxt as u

import coordinax.ops as cxo


class _QuantityFieldOperator(cxo.AbstractOperator):
    """An operator holding a bare quantity, with a quantity default.

    None of the shipped operators has this shape, which is why the bug this
    guards reached a release: `__pdoc__` drops fields equal to their class
    default, and comparing two quantities yields a dimensionless *quantity*
    under unxt v2, not a bare array.
    """

    omega: u.AbstractQuantity = u.Q(1.0, "1/Gyr")

    @property
    def is_inertial(self) -> bool:
        return True

    @property
    def inverse(self) -> "_QuantityFieldOperator":
        return _QuantityFieldOperator(omega=-self.omega)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def test_repr_with_a_quantity_field_equal_to_its_default() -> None:
    """`jax.numpy.all` rejects a quantity; the filter must tolerate one."""
    assert "_QuantityFieldOperator" in repr(_QuantityFieldOperator())


def test_repr_with_a_quantity_field_differing_from_its_default() -> None:
    """The other branch of the same comparison."""
    op = _QuantityFieldOperator(omega=u.Q(2.0, "1/Gyr"))
    assert "_QuantityFieldOperator" in repr(op)
