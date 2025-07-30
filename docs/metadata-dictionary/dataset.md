# Dataset

Collection of signals representing a complete monitoring dataset.
    
    A Dataset groups multiple signals that are collected together as part of
    a monitoring project or analysis workflow. It provides project-level
    metadata and enables coordinated processing operations across multiple
    parameters.
    
    Datasets support cross-signal processing operations and maintain consistent
    naming conventions across all contained signals. They provide the highest
    level of organization for environmental monitoring data with complete
    metadata preservation and serialization capabilities.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `created_on` | `datetime` | ✗ | `Factory: now()` | Timestamp when this Dataset was created |
| `last_updated` | `datetime` | ✗ | `Factory: now()` | Timestamp of the most recent modification to this Dataset |
| `name` | `str` | ✓ | `—` | Name identifying this dataset |
| `description` | `None` | ✗ | `None` | Detailed description of the dataset contents and purpose |
| `owner` | `None` | ✗ | `None` | Person or organization responsible for this dataset |
| `signals` | `dict` | ✓ | `—` | Dictionary mapping signal names to Signal objects in this dataset |
| `purpose` | `None` | ✗ | `None` | Purpose or objective of this dataset (e.g., 'compliance_monitoring', 'research') |
| `project` | `None` | ✗ | `None` | Project or study name associated with this dataset |

## Detailed Field Descriptions

### created_on

**Type:** `datetime`
**Required:** No
**Default:** Factory: now()

Timestamp when this Dataset was created

### last_updated

**Type:** `datetime`
**Required:** No
**Default:** Factory: now()

Timestamp of the most recent modification to this Dataset

### name

**Type:** `str`
**Required:** Yes

Name identifying this dataset

### description

**Type:** `None`
**Required:** No
**Default:** None

Detailed description of the dataset contents and purpose

### owner

**Type:** `None`
**Required:** No
**Default:** None

Person or organization responsible for this dataset

### signals

**Type:** `dict`
**Required:** Yes

Dictionary mapping signal names to Signal objects in this dataset

### purpose

**Type:** `None`
**Required:** No
**Default:** None

Purpose or objective of this dataset (e.g., 'compliance_monitoring', 'research')

### project

**Type:** `None`
**Required:** No
**Default:** None

Project or study name associated with this dataset

## Usage Example

```python
from meteaudata.types import Dataset

# Create a Dataset instance
dataset = Dataset(
    name="river_monitoring_2024",
    description="Continuous water quality monitoring",
    owner="Environmental Team",
    signals={
        "temperature": temp_signal,
        "dissolved_oxygen": do_signal
    },
    project="water_quality_assessment"
)
```
