# Contributing

# Contributing to metEAUdata

Thank you for your interest in contributing to metEAUdata! We welcome
contributions that help improve environmental time series data processing
and analysis.

## Getting Started

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Submit a pull request to the `main` branch

For major changes, please open an issue first to discuss your proposed
changes.

## Types of Contributions

We accept the following types of pull requests:

- **Bug fixes** - Help us improve reliability and correctness
- **New transformation functions** - Add processing capabilities that
conform to `SignalTransformFunctionProtocol` or
`DatasetTransformFunctionProtocol`
- **Addition of metadata attributes** - Enhance data provenance and
processing history tracking

## Development Guidelines

- Follow existing code style and patterns
- Add appropriate type hints and docstrings
- Ensure all tests pass before submitting
- Update documentation for new features

### Documentation Requirements

When contributing new features, please update the relevant documentation templates to showcase your additions:

#### New Metadata Attributes
- **Add working examples** showing how objects are instantiated with new attributes
- **Update relevant templates** in `docs/*/template.md` files where the attributes are used
- **Include default values** and explain their purpose in context

#### New Processing Functions  
- **Add function examples** to the appropriate API reference templates:
  - Single-parameter functions → `docs/api-reference/processing/univariate_template.md`
  - Multi-parameter functions → `docs/api-reference/processing/multivariate_template.md`
- **Include complete workflows** showing the function in realistic processing pipelines
- **Document parameters** and their effects on data processing
- **Show before/after examples** with actual data transformations

#### General Documentation Guidelines
- **Use executable code blocks** (`python exec`) where possible to ensure examples stay current
- **Include real outputs** by running examples during documentation build
- **Follow existing patterns** in template files for consistency
- **Test all examples** to ensure they execute without errors

The documentation system uses template files (`*_template.md`) that generate final documentation with executable code blocks. This ensures all examples are tested and current with each release.

## Questions?

Open an issue if you have questions about contributing or need help getting
started.
