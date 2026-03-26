# Synthetic Enterprise Simulator

Synthetic Enterprise Simulator generates deterministic synthetic enterprise data for training and evaluating models that need to identify relevant records across email, Slack, Microsoft Teams, and Salesforce CRM.

The system is built around one rule: messages and CRM rows are rendered from an underlying simulated company state, not invented independently. That makes records traceable, cross-system consistent, and suitable for relevance extraction, linking, and hard-negative training.

## Project Goals

- Simulate a believable enterprise before generating communications.
- Produce both relevant and non-relevant records with heavy noise.
- Keep generation deterministic through explicit seeds and chunk seeds.
- Support source-specific exports, streaming generation, manifests, and quality evaluation.
- Maintain a modular Python codebase with strict schemas and TDD-driven growth.

## Quick Start

Install the project in editable mode:

```bash
make install
```

Run the full test suite:

```bash
make test
make lint
make typecheck
```

Generate a small mixed dataset:

```bash
synthetic-enterprise generate-all \
  --seed 17 \
  --output-root out/small \
  --account-count 4 \
  --min-employees 30 \
  --max-employees 30 \
  --email-count 50 \
  --slack-count 50 \
  --teams-count 50 \
  --crm-count 50
```

This writes:

- `out/small/company_state.json`
- `out/small/source=email/...`
- `out/small/source=slack/...`
- `out/small/source=teams/...`
- `out/small/source=salesforce/...`

Each source partition contains Parquet files plus a `manifest.json` with seed, schema version, row count, chunk count, export formats, and config.

## Architecture Overview

High-level generation flow:

```text
seed
  -> GeneratorContext
  -> CompanyBuilder
  -> EnterpriseGraph
  -> CrossSystemRenderer + source renderers
  -> grounded source rows
  -> dataset writers
  -> parquet/csv/jsonl + manifest
  -> quality report / gold set / training formats
```

Core package areas:

- `synthetic_enterprise/domain`: enterprise entities and relationship validation
- `synthetic_enterprise/generation`: seeded context, company simulation, cross-system rendering, pipeline orchestration
- `synthetic_enterprise/contracts/records`: row schemas for email, Slack, Teams, and Salesforce
- `synthetic_enterprise/labeling`: taxonomy, grounding, provenance, hard-negative logic
- `synthetic_enterprise/storage`: dataframe conversion and export writers
- `synthetic_enterprise/evaluation`: quality reports, gold subsets, training-ready exports
- `synthetic_enterprise/app/cli.py`: command-line entry points

Detailed notes: [docs/architecture.md](/Users/teegsontech/synthetic_enterprise_generator/docs/architecture.md)

## Data Model Overview

The simulator builds an `EnterpriseGraph` containing explicit relationships among:

- `Company`
- `Department`
- `Employee`
- `CustomerAccount`
- `Contact`
- `Opportunity`
- `Event`
- `Product`
- `TicketIssue`
- `Campaign`
- `MessageEnvelope`
- `CRMActivity`

Source rows then reference those entities through stable ids such as:

- `account_id`
- `event_id`
- `ticket_id`
- `opportunity_id`
- `linked_account_id`
- `linked_event_id`
- `campaign_id`

All timestamps are timezone-aware. All major schemas support dict and JSON serialization.

Detailed notes: [docs/data-model.md](/Users/teegsontech/synthetic_enterprise_generator/docs/data-model.md)

## How Relevance Is Grounded

Every generated row carries:

- `primary_category`
- `is_relevant`
- `relevance_reason`
- `provenance`

`provenance` links the row back to a simulated object such as an `event`, `ticket`, `opportunity`, or `campaign`. Relevant rows require strong provenance. Hard negatives may carry weak provenance. Generic noise can have `null` provenance.

Example:

```json
{
  "source_system": "email",
  "primary_category": "blocker",
  "is_relevant": true,
  "event_id": "event_a7d9d13ba78691ce",
  "ticket_id": "ticket_issue_9689393d5af6f8a1",
  "relevance_reason": "Constraint is grounded in ticket ticket_issue_9689393d5af6f8a1: Open issue 'Users need faster resolution on integration errors.' is constraining the next step.",
  "provenance": {
    "object_type": "ticket",
    "object_id": "ticket_issue_9689393d5af6f8a1",
    "strength": "strong",
    "explanation": "Open issue 'Users need faster resolution on integration errors.' is constraining the next step."
  }
}
```

Detailed notes: [docs/grounding-and-noise.md](/Users/teegsontech/synthetic_enterprise_generator/docs/grounding-and-noise.md)

## How Noise Is Generated

Noise is not only random filler. The system currently generates:

- generic channel noise such as greetings, status updates, scheduling-only coordination, and admin chatter
- low-value CRM artifacts such as stale opportunities, duplicate tasks, and irrelevant marketing records
- hard negatives that overlap lexically with relevant records but remain non-relevant

Important controls:

- `noise_ratio`
- `hard_negative_ratio`
- `cross_system_ratio`
- `chunk_size`
- `generation_chunk_size`

## Run A Small Dataset

Generate one company only:

```bash
synthetic-enterprise simulate-company \
  --seed 17 \
  --output-root out/company \
  --account-count 4 \
  --min-employees 30 \
  --max-employees 30
```

Generate source-specific slices:

```bash
synthetic-enterprise generate-email --seed 17 --output-root out/email --email-count 100
synthetic-enterprise generate-slack --seed 17 --output-root out/slack --slack-count 100
synthetic-enterprise generate-teams --seed 17 --output-root out/teams --teams-count 100
synthetic-enterprise generate-crm --seed 17 --output-root out/crm --crm-count 100
```

Generate a full small dataset:

```bash
synthetic-enterprise generate-all \
  --seed 17 \
  --output-root out/small \
  --account-count 20 \
  --min-employees 30 \
  --max-employees 30 \
  --email-count 200 \
  --slack-count 200 \
  --teams-count 200 \
  --crm-count 200
```

Detailed operating notes: [docs/operations.md](/Users/teegsontech/synthetic_enterprise_generator/docs/operations.md)

## Scale To Large Datasets

For larger runs, use streaming generation:

```bash
synthetic-enterprise generate-all \
  --seed 17 \
  --output-root out/large \
  --account-count 500 \
  --email-count 500000 \
  --slack-count 500000 \
  --teams-count 500000 \
  --crm-count 500000 \
  --stream \
  --generation-chunk-size 50000 \
  --chunk-size 50000
```

This keeps generation memory-aware by producing deterministic source chunks and writing them incrementally rather than holding all rows in memory.

## Evaluate Output Quality

Create a quality report:

```bash
synthetic-enterprise quality-report \
  --input-root out/small \
  --output-path out/small/quality_report.json
```

The report includes:

- relevance rate
- category distribution
- noise category distribution
- thread depth distribution
- average message length by source
- cross-system linkage rate
- hard-negative rate
- duplicate rate
- lexical diversity
- entity coverage

Build a smaller clean benchmark subset:

```bash
synthetic-enterprise build-gold-set \
  --input-root out/small \
  --output-root out/gold \
  --relevant-category buying_signal \
  --relevant-category event_attendance \
  --relevant-category blocker \
  --samples-per-relevant-category 10 \
  --hard-negative-count 20 \
  --min-cross-system-examples 5
```

## Examples

Relevant row:

```json
{
  "source_system": "email",
  "subject": "Confirmed attendees for Quarterly review with Davis-Andrews",
  "primary_category": "event_attendance",
  "is_relevant": true,
  "event_id": "event_a7d9d13ba78691ce",
  "relevance_reason": "Attendance is grounded in event event_a7d9d13ba78691ce: Confirmed attendee list is attached to the customer review."
}
```

Noisy row:

```json
{
  "source_system": "slack",
  "channel_name": "#watercooler",
  "body": "morning all",
  "primary_category": "greetings",
  "is_relevant": false,
  "relevance_reason": "Greeting only.",
  "provenance": null
}
```

Hard negative:

```json
{
  "source_system": "teams",
  "body": "Agenda addendum:\n- move review logistics to next week after travel rebooking\n- no attendee confirmation is needed yet",
  "primary_category": "scheduling_only",
  "is_relevant": false,
  "relevance_reason": "Calendar logistics are weakly tied to event event_1a1bdb3c1ff8f608: Meeting logistics mention the review without attendance or follow-up action."
}
```

Cross-system consistency example:

- Email: attendee confirmation for `event_a7d9d13ba78691ce`
- Teams: meeting coordination for `event_a7d9d13ba78691ce`
- Salesforce: CRM event and campaign-member records for `event_a7d9d13ba78691ce`
- Slack: internal buying-signal thread linked to the same account and opportunity around that event

More complete examples: [docs/examples.md](/Users/teegsontech/synthetic_enterprise_generator/docs/examples.md)

## Limitations And Ethics

- The generator aims for realism, not truth. It is useful for model development, not for measuring real-world business frequency.
- Synthetic corpora can still encode modeling assumptions, taxonomy bias, and style artifacts.
- The system is designed to learn structural features of enterprise communication, not to recreate proprietary corpora.
- Human review is still needed before treating any generated benchmark as a production-grade evaluation set.

Detailed notes: [docs/limitations.md](/Users/teegsontech/synthetic_enterprise_generator/docs/limitations.md)

## Docs Map

- [docs/README.md](/Users/teegsontech/synthetic_enterprise_generator/docs/README.md)
- [docs/architecture.md](/Users/teegsontech/synthetic_enterprise_generator/docs/architecture.md)
- [docs/data-model.md](/Users/teegsontech/synthetic_enterprise_generator/docs/data-model.md)
- [docs/grounding-and-noise.md](/Users/teegsontech/synthetic_enterprise_generator/docs/grounding-and-noise.md)
- [docs/operations.md](/Users/teegsontech/synthetic_enterprise_generator/docs/operations.md)
- [docs/examples.md](/Users/teegsontech/synthetic_enterprise_generator/docs/examples.md)
- [docs/limitations.md](/Users/teegsontech/synthetic_enterprise_generator/docs/limitations.md)
