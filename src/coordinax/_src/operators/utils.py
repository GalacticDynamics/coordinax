"""Core operator API functions.

This module defines helpers for operator implementations.
"""

__all__: tuple[str, ...] = ()

import equinox as eqx

import coordinax._src.roles as cxr

# =============================================================================
# Helper to enforce role requirement for CsDict


def _require_role_for_pdict(role: cxr.AbstractRole | None) -> cxr.AbstractRole:
    """Raise TypeError if role is None for CsDict inputs.

    Parameters
    ----------
    role : AbstractRole | None
        The role argument passed to ``apply_op``.

    Returns
    -------
    AbstractRole
        The validated role (unchanged if not None).

    Raises
    ------
    TypeError
        If ``role`` is None.

    """
    return eqx.error_if(
        role,
        role is None,
        "`apply_op` on CsDict requires explicit `role=...` "
        "(e.g. `role=cx.roles.point`).",
    )
