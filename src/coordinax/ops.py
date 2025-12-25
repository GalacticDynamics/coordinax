"""Operators to effectuate transformations.

Operators are transformations that can be applied to various objects in
`coordinax`, such as vectors and coordinates. They encapsulate operations like
identity transformations and translations (additions). They can be combined and
simplified. Most powerfully, they can be time dependent.

Let's work through the built-in operations.

>>> import jax
>>> import jax.numpy as jnp
>>> import unxt as u
>>> import coordinax.vecs as cxv
>>> import coordinax.ops as cxo
>>> import coordinax.frames as cxf

>>> x = jnp.asarray([1.0, 2.0, 3.0])
>>> q = u.Quantity(x, "km")
>>> vec = cxv.CartesianPos3D.from_(q)
>>> vel = cxv.CartesianVel3D.from_([4, 5, 6], "km/s")
>>> frame = cxf.ICRS()
>>> coord = cxf.Coordinate({"length": vec, "speed": vel}, frame=frame)

# Identity Operator

>>> identity_op = cxo.Identity()

An :class:`coordinax.ops.Identity` operator cannot be further simplified:

>>> cxo.simplify(identity_op) is identity_op
True

An :class:`coordinax.ops.Identity` operator leaves the input unchanged:

>>> identity_op(None, x) is x
True
>>> identity_op(None, q) is q
True
>>> identity_op(None, vec) is vec
True
>>> identity_op(None, vel) is vel
True
>>> identity_op(None, coord) is coord
True

The inverse of an :class:`coordinax.ops.Identity` operator is itself:

>>> identity_op.inverse is identity_op
True


# Add Operator

The addition operator can be simplified if the delta is zero.

>>> add_op = cxo.Add(jnp.array([0.0, 0.0, 0.0]))
>>> simplified_add = cxo.simplify(add_op)
>>> isinstance(simplified_add, cxo.Identity)
True

If the delta is non-zero, the operator cannot be simplified:

>>> add_op = cxo.Add(jnp.array([1.0, 2.0, 3.0]))
>>> simplified_add = cxo.simplify(add_op)
>>> simplified_add == add_op
Array(True, dtype=bool)

An :class:`coordinax.ops.Add` operator translates the input by a fixed amount.
Addition rules depend on the types of the parameters and input:

>>> add_op_x = cxo.Add(jnp.array([1.0, 1.0, 1.0]))
>>> add_op_x(x)
Array([2., 3., 4.], dtype=float32)

>>> add_op_q = cxo.Add(u.Quantity([1.0, 1.0, 1.0], "km"))
>>> add_op_q(q)
Quantity(Array([2., 3., 4.], dtype=float32), unit='km')

>>> shift_vec = cxv.CartesianPos3D.from_([1.0, 1.0, 1.0], "km")
>>> add_op_vec = cxo.Add(shift_vec)
>>> print(add_op_vec(vec))
<CartesianPos3D: (x, y, z) [km]
    [2. 3. 4.]>

>>> print(add_op_vec(coord))  # a purely spatial translation
Coordinate( {
    'length': <CartesianPos3D: (x, y, z) [km]
        [2. 3. 4.]>,
    'speed': <CartesianVel3D: (x, y, z) [km / s]
        [4 5 6]>
    },
    frame=ICRS()
)

Operators can be time dependent by making their parameters functions of time:

>>> op = cxo.Add(lambda t: u.Quantity(t.ustrip("yr"), "km"))
>>> op(u.Quantity(0, "yr"), q)
Quantity(Array([1., 2., 3.], dtype=float32), unit='km')

>>> jax.vmap(op, (0, None))(u.Quantity([0, 2, 4], "yr"), q)
Quantity(Array([[1., 2., 3.],
                [3., 4., 5.],
                [5., 6., 7.]], dtype=float32), unit='km')

The inverse of an :class:`coordinax.ops.Add` operator translates by the negative
of the original delta:

>>> print(add_op_vec.inverse)
Add(<CartesianPos3D: (x, y, z) [km]
        [-1. -1. -1.]>)


# Rotate Operator

The rotation operator can be initialized from Euler angles:

>>> R = cxo.Rotate.from_euler("z", u.Quantity(90, "deg"))
>>> print(R.R.round(2))
[[ 0. -1.  0.]
 [ 1.  0.  0.]
 [ 0.  0.  1.]]

Rotations cannot be simplfied unless they are identity rotations.

>>> simplified_rot = cxo.simplify(R)
>>> simplified_rot == R
Array(True, dtype=bool)

>>> cxo.simplify(cxo.Rotate(jnp.eye(3)))  # identity rotation
Identity()

An :class:`coordinax.ops.Rotate` operator rotates the input vector:

>>> print(R(x).round(2))
[-2.  1.  3.]

>>> print(R(q).round(2))
Quantity['length']([-2.,  1.,  3.], unit='km')

>>> print(R(vec))
<CartesianPos3D: (x, y, z) [km]
    [-2.  1.  3.]>

# >>> print(R(coord))  # TODO: support coordinate rotation

The inverse of an :class:`coordinax.ops.Rotate` operator is the transpose of the
original rotation matrix:

>>> R_inv = R.inverse
>>> print(R_inv.R.round(2))
[[ 0.  1.  0.]
 [-1.  0.  0.]
 [ 0.  0.  1.]]


# Combining Operators

## Pipe Operator

Operators can be combined using the pipe (`|`) operator, which results in a
:class:`coordinax.ops.Pipe` operator.

>>> combined_op = add_op_vec | R
>>> print(combined_op)
Pipe((
    Add(<CartesianPos3D: (x, y, z) [km]
            [1. 1. 1.]>),
    Rotate([[ 0.         -0.99999994  0.        ]
            [ 0.99999994  0.          0.        ]
            [ 0.          0.          0.99999994]])
))

Applying the combined operator is equivalent to applying each component in sequence:

>>> result_combined = combined_op(vec)
>>> print(result_combined)
<CartesianPos3D: (x, y, z) [km]
    [-3.  2.  4.]>

>>> result_sequential = R(add_op_vec(vec))
>>> result_combined == result_sequential
Array(True, dtype=bool)

Combined operators can be simplified if possible:

>>> print(cxo.simplify(identity_op | add_op_vec))
Add(<CartesianPos3D: (x, y, z) [km]
        [1. 1. 1.]>)


Combined operators have inverses that reverse the order of operations:

>>> combined_inv = combined_op.inverse
>>> combined_inv
Pipe((
    Rotate(R=f32[3,3]),
    Add( delta=CartesianPos3D( ... ) )
))

Any component operator can be time dependent, allowing for complex,
time-varying transformations.

>>> time_dep_op = cxo.Add(lambda t: vec * t.ustrip("yr")) | R
>>> print(time_dep_op(u.Quantity(1, "yr"), vec))
<CartesianPos3D: (x, y, z) [km]
    [-4.  2.  6.]>


## Galilean Operator

"""
# pylint: disable=unused-wildcard-import,wildcard-import

from ._src.operators import *  # noqa: F403
from ._src.operators import __all__  # noqa: F401
