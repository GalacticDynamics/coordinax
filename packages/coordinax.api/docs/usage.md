# Usage Guide

This guide demonstrates how to use {mod}`coordinax.api` to implement custom vector types and coordinate transformations that integrate with the `coordinax` ecosystem.

The {mod}`coordinax.api` package uses [plum-dispatch](https://github.com/beartype/plum) for multiple dispatch. This allows different implementations of the same function based on the types of the arguments.

## Best Practices

1. **Type hints**: Always use proper type hints for dispatch to work correctly
2. **Documentation**: Document what your dispatch expects and returns
3. **Error handling**: Raise clear errors for unsupported conversions
4. **Testing**: Test your dispatches with various input types

## Next Steps

- See the {doc}`API Reference </packages/coordinax.api/api>` for complete documentation
- Check out the [coordinax documentation](https://coordinax.readthedocs.io/) for concrete implementations
- Review the plum-dispatch documentation for advanced dispatch patterns
