# Architecture

## Overview

The simulator is organized around deterministic state first and rendered records second.

```text
seed
  -> GeneratorConfig
  -> GeneratorContext
  -> CompanyBuilder
  -> EnterpriseGraph
  -> CrossSystemRenderer and source renderers
  -> grounded records
  -> dataset writers and manifests
  -> evaluation and training exports
```

That separation matters:

- domain modules own business truth
- source renderers own channel-specific phrasing and row fields
- labeling modules own taxonomy, provenance, and relevance reasons
- storage modules own dataframe conversion and export
- evaluation modules own quality metrics and benchmark subsets

## Main Packages

### `synthetic_enterprise/domain`

Validated enterprise entities and the `EnterpriseGraph` aggregate.

Responsibilities:

- strict schema validation
- explicit relationships
- stable ids
- JSON and dict serialization

### `synthetic_enterprise/generation`

Seeded orchestration and simulation logic.

Responsibilities:

- generator config and seeded context
- company simulation
- chunk planning and streaming generation
- cross-system event rendering

### `synthetic_enterprise/contracts/records`

Source-specific row contracts.

Responsibilities:

- email, Slack, Teams, and Salesforce row schemas
- required fields and validation rules
- dataframe export helpers

### `synthetic_enterprise/labeling`

Taxonomy and label grounding.

Responsibilities:

- relevant vs noise categories
- provenance validation
- rationale generation
- hard-negative handling

### `synthetic_enterprise/storage`

Dataset writers and manifests.

Responsibilities:

- pandas conversion
- parquet and optional CSV output
- chunk-aware writing
- schema metadata

### `synthetic_enterprise/evaluation`

Post-generation quality and benchmark tooling.

Responsibilities:

- quality reports
- gold subset construction
- training-ready export formats

## Determinism Model

Determinism is enforced through `GeneratorContext`.

- one root seed drives the run
- derived namespace seeds are used for sub-generators and chunks
- faker and random are seeded from the same deterministic context
- streaming generation derives stable chunk seeds from the root seed

This means a fixed seed and config should reproduce the same enterprise state, row ids, and chunk layout.

## Cross-System Rendering

Cross-system consistency is handled before source-specific noise is added.

1. Build an enterprise graph.
2. Select candidate account-linked events and opportunities.
3. Render coherent bundles across email, Slack, Teams, and Salesforce.
4. Top up each source with deterministic noise and hard negatives.
5. Export each source independently while preserving shared ids.

This keeps facts aligned even when wording differs by channel.

## Export Boundary

The storage layer treats each source independently but writes a consistent layout:

```text
output_root/
  company_state.json
  source=email/
    manifest.json
    parquet/
      part-00000.parquet
  source=slack/
  source=teams/
  source=salesforce/
```

The manifest is the contract between generation and downstream consumers. It records:

- source name
- schema version
- seed
- config
- row count
- chunk count
- columns
- output files

## Design Tradeoffs

- Realism is favored over maximal randomness.
- Schema strictness is favored over permissive row generation.
- Cross-system linkage is explicit, not inferred after the fact.
- Streaming support is designed into the pipeline instead of being bolted on later.
