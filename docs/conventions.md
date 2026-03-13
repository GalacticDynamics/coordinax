# 📜 Conventions

## Naming Conventions

{mod}`coordinax` uses a few conventions to make the code more readable and to avoid verbosity. Many of these are also found in the [Glossary](glossary.md).

- `Abstract...`: a class that is not meant to be instantiated directly, but rather to be subclassed. Abstract classes are prefixed with 'Abstract'. Concrete (or 'final') classes are not so prefixed. As a further rule, no abstract class inherits from a concrete class and no concrete class inherits from any other concrete class.
- `PhysDisp`: a shorthand for "position", used in class names for concision.
- `PhysVel`: a shorthand for "velocity", used in class names for concision.
- `PhysAcc`: a shorthand for "acceleration", used in class names for concision.

## Functional vs Object-Oriented APIs

As {mod}`JAX` is function-oriented, but Python is generally object-oriented, {mod}`coordinax` provides both functional and object-oriented APIs. The functional APIs are the primary APIs, but the object-oriented APIs are easy to use and call the functional APIs, so lose none of the power.

## Multiple Dispatch

`coordinax` uses [multiple dispatch](https://beartype.github.io/plum/) to hook into `quax`'s flexible and extensible system to enable custom array-ish objects, like {class}`~unxt.quantity.Quantity`, in {mod}`JAX`. Also, {mod}`coordinax` uses multiple dispatch to enable deep interoperability between {mod}`coordinax` and other libraries, like {mod}`astropy` (and anything user-defined).

For more information on multiple dispatch, see the [plum documentation](https://beartype.github.io/plum/).
