# DataProvenance

!!! abstract "Usage Documentation"
    [Models](../concepts/models.md)

A base class for creating Pydantic models.

Attributes:
    __class_vars__: The names of the class variables defined on the model.
    __private_attributes__: Metadata about the private attributes of the model.
    __signature__: The synthesized `__init__` [`Signature`][inspect.Signature] of the model.

    __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
    __pydantic_core_schema__: The core schema of the model.
    __pydantic_custom_init__: Whether the model has a custom `__init__` function.
    __pydantic_decorators__: Metadata containing the decorators defined on the model.
        This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
    __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
        __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
    __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
    __pydantic_post_init__: The name of the post-init method for the model, if defined.
    __pydantic_root_model__: Whether the model is a [`RootModel`][pydantic.root_model.RootModel].
    __pydantic_serializer__: The `pydantic-core` `SchemaSerializer` used to dump instances of the model.
    __pydantic_validator__: The `pydantic-core` `SchemaValidator` used to validate instances of the model.

    __pydantic_fields__: A dictionary of field names and their corresponding [`FieldInfo`][pydantic.fields.FieldInfo] objects.
    __pydantic_computed_fields__: A dictionary of computed field names and their corresponding [`ComputedFieldInfo`][pydantic.fields.ComputedFieldInfo] objects.

    __pydantic_extra__: A dictionary containing extra values, if [`extra`][pydantic.config.ConfigDict.extra]
        is set to `'allow'`.
    __pydantic_fields_set__: The names of fields explicitly set during instantiation.
    __pydantic_private__: Values of private attributes set on the model instance.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source_repository` | `None` | ✗ | `—` | No description provided |
| `project` | `None` | ✗ | `—` | No description provided |
| `location` | `None` | ✗ | `—` | No description provided |
| `equipment` | `None` | ✗ | `—` | No description provided |
| `parameter` | `None` | ✗ | `—` | No description provided |
| `purpose` | `None` | ✗ | `—` | No description provided |
| `metadata_id` | `None` | ✗ | `—` | No description provided |

## Detailed Field Descriptions

### source_repository

**Type:** `None`
**Required:** No

No description provided

### project

**Type:** `None`
**Required:** No

No description provided

### location

**Type:** `None`
**Required:** No

No description provided

### equipment

**Type:** `None`
**Required:** No

No description provided

### parameter

**Type:** `None`
**Required:** No

No description provided

### purpose

**Type:** `None`
**Required:** No

No description provided

### metadata_id

**Type:** `None`
**Required:** No

No description provided

## Usage Example

```python
from meteaudata.types import DataProvenance

# Create a DataProvenance instance
instance = DataProvenance(
)
```
