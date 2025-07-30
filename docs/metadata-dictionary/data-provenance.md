# DataProvenance

Information about the source and context of time series data.
    
    This class captures essential metadata about where time series data originated,
    including the source repository, project context, physical location, equipment
    used, and the measured parameter. This information is crucial for data
    traceability and understanding measurement context in environmental monitoring.
    
    Provenance information enables users to assess data quality, understand
    measurement conditions, and make informed decisions about data usage in
    analysis and modeling workflows.

## Field Definitions

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source_repository` | `None` | ✗ | `None` | Name or identifier of the data repository or database |
| `project` | `None` | ✗ | `None` | Project name or identifier under which data was collected |
| `location` | `None` | ✗ | `None` | Physical location where measurements were taken (e.g., 'Site_A', 'Influent_Tank_1') |
| `equipment` | `None` | ✗ | `None` | Equipment or instrument used for data collection (e.g., 'pH_probe_001', 'flow_meter') |
| `parameter` | `None` | ✗ | `None` | Physical/chemical parameter being measured (e.g., 'temperature', 'dissolved_oxygen', 'TSS') |
| `purpose` | `None` | ✗ | `None` | Purpose or context of the measurement (e.g., 'regulatory_compliance', 'process_optimization') |
| `metadata_id` | `None` | ✗ | `None` | Unique identifier for linking to external metadata systems |

## Detailed Field Descriptions

### source_repository

**Type:** `None`
**Required:** No
**Default:** None

Name or identifier of the data repository or database

### project

**Type:** `None`
**Required:** No
**Default:** None

Project name or identifier under which data was collected

### location

**Type:** `None`
**Required:** No
**Default:** None

Physical location where measurements were taken (e.g., 'Site_A', 'Influent_Tank_1')

### equipment

**Type:** `None`
**Required:** No
**Default:** None

Equipment or instrument used for data collection (e.g., 'pH_probe_001', 'flow_meter')

### parameter

**Type:** `None`
**Required:** No
**Default:** None

Physical/chemical parameter being measured (e.g., 'temperature', 'dissolved_oxygen', 'TSS')

### purpose

**Type:** `None`
**Required:** No
**Default:** None

Purpose or context of the measurement (e.g., 'regulatory_compliance', 'process_optimization')

### metadata_id

**Type:** `None`
**Required:** No
**Default:** None

Unique identifier for linking to external metadata systems

## Usage Example

```python
from meteaudata.types import DataProvenance

# Create a DataProvenance instance
provenance = DataProvenance(
    source_repository="station_database",
    project="water_quality_monitoring",
    location="river_site_A",
    equipment="multiparameter_probe",
    parameter="dissolved_oxygen",
    purpose="compliance_monitoring"
)
```
