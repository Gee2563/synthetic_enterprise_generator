# Phase 2 Improvements

Phase 2 extends the original deterministic simulator with deeper scenario state, more varied language, richer source behavior, harder negatives, and stronger realism benchmarking.

The design constraint did not change: relevant rows still need explicit provenance, ids remain deterministic under seed, and export contracts stay stable.

## Scenario Engine V2

Phase 1 mostly rendered isolated business facts. Phase 2 introduces lifecycle-based scenario arcs that evolve over time.

Supported scenario families currently include:

- `pre_sales_discovery`
- `procurement_delay`
- `pilot_success`
- `rollout_risk`
- `support_escalation`
- `feature_gap_review`
- `renewal_risk`
- `executive_sponsor_review`
- `event_invite_to_attendance`
- `champion_departure`
- `stakeholder_expansion`
- `contract_redline_delay`

Each scenario now carries:

- ordered lifecycle stages
- timestamped transitions
- entity-level state transitions
- downstream communication triggers
- category-to-provenance grounding

This is what lets event attendance, blockers, churn risk, and decision-maker signals emerge from evolving business state instead of static labels.

## Company Language Profiles

Different companies now render with distinct lexical habits and channel language.

Current profile families include:

- `enterprise_b2b_formal`
- `fast_moving_startup`
- `field_ops_heavy`
- `security_sensitive_enterprise`
- `product_led_saas`
- `consulting_led_org`

These profiles influence:

- vocabulary and jargon
- abbreviation density
- directness
- email formality
- Slack informality
- Teams meeting language
- CRM note style
- team aliases and internal code names

The same underlying event can therefore produce different wording across different simulated enterprises while preserving the same linked facts.

## Persona Voice Model

Phase 2 also differentiates voice by sender persona instead of only by source.

Current persona rules vary:

- brevity
- assertiveness
- hedging
- technical detail
- commercial wording
- executive summarization
- customer-facing phrasing

Examples of supported persona distinctions:

- AE vs support engineer
- executive vs IC
- internal vs external
- CSM vs implementation manager
- operations lead vs product manager

This keeps generated text more realistic without relying on unconstrained free-form generation.

## Compositional Template System

Phase 1 message rendering relied more heavily on monolithic template bodies. Phase 2 moved core email generation to a compositional system that can vary:

- openings
- core clauses
- context blocks
- objections
- action asks
- closings
- signatures
- disclaimers
- prior-thread references
- sentence order

That structure is then combined with a controlled paraphrase layer, so surface form can vary while linked facts, dates, and provenance remain stable.

## Source-Specific Realism

Phase 2 deepens channel differentiation instead of treating all sources as generic text carriers.

### Email

- forwarded chains
- reply-all behavior
- signatures and disclaimers
- external vs internal tone
- subject drift in longer threads

### Slack

- reactions and shorthand
- quick clarification loops
- terse fragments
- channel-specific tone
- lightweight acknowledgements

### Teams

- meeting-linked follow-ups
- file references
- structured recap messages
- task-oriented coordination
- more project-oriented language

### Salesforce

- terse notes
- low-quality duplicate updates
- stale ownership or stage artifacts
- admin/event/campaign records
- structured plus semi-structured text fields

## Messy-Data Model

Phase 2 adds bounded messiness to reduce synthetic regularity while keeping rows schema-valid.

Examples now supported:

- missing CRM fields
- delayed note entry
- contradictory notes
- reopened cases
- incomplete attendee capture
- outdated opportunity stage sync
- stale ownership snapshots
- missing or duplicate follow-up task capture

Messiness is explicit enough to be measurable, but not so destructive that grounding or export validity breaks.

## Cross-System Lag And Partial Visibility

Facts no longer appear everywhere at the same time.

Phase 2 supports:

- Slack discussing an issue before CRM is updated
- email confirming attendance before campaign-member status lands in CRM
- Salesforce notes lagging behind Teams coordination
- some linked events appearing in only a subset of systems

This makes cross-system linkage more realistic while preserving shared ids and fact consistency.

## New Quality Metrics

Phase 2 extends quality reporting with realism-focused metrics:

- `duplicate_rate_by_source`
- `lexical_diversity_by_source`
- `lexical_diversity_by_company`
- `scenario_coverage`
- `source_style_separation_proxy`
- `hard_negative_difficulty_proxy`
- `noise_family_entropy`
- `temporal_lag_distribution`
- `account_profile_coverage`
- `messy_data_rate`

These metrics are useful both for dataset tuning and for regression protection as Phase 2 evolves.

## How To Benchmark Realism Improvements

There are now three practical layers for realism benchmarking.

### 1. Generate a baseline Phase 1-style dataset

Use the regular pipeline:

```python
from pathlib import Path

from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import DatasetTargets, GenerationPipeline

context = GeneratorContext(
    seed=17,
    config=GeneratorConfig(
        company_size=CompanySizeConfig(min_employees=30, max_employees=30),
        noise_ratio=0.85,
        hard_negative_ratio=0.25,
        cross_system_ratio=0.5,
    ),
)
pipeline = GenerationPipeline(context=context)
baseline = pipeline.generate_dataset(
    targets=DatasetTargets(
        account_count=20,
        email_count=200,
        slack_count=200,
        teams_count=200,
        salesforce_count=200,
    ),
    destination_root=Path("out/phase1"),
)
baseline_report = DatasetQualityEvaluator(
    enterprise=baseline.enterprise,
    context=context,
).evaluate(
    {
        "email": [row.to_dict() for row in baseline.email_records],
        "slack": [row.to_dict() for row in baseline.slack_records],
        "teams": [row.to_dict() for row in baseline.teams_records],
        "salesforce": [row.to_dict() for row in baseline.salesforce_records],
    }
)
```

### 2. Generate the dedicated Phase 2 benchmark

```python
from pathlib import Path

from synthetic_enterprise.generation.pipeline import (
    DatasetTargets,
    GenerationPipeline,
    Phase2BenchmarkTargets,
)

phase2 = pipeline.generate_phase2_benchmark(
    targets=Phase2BenchmarkTargets(
        company_count=3,
        per_company_targets=DatasetTargets(
            account_count=12,
            email_count=18,
            slack_count=18,
            teams_count=18,
            salesforce_count=18,
            chunk_size=9,
        ),
    ),
    destination_root=Path("out/phase2"),
)
```

### 3. Compare reports

For direct Phase 1 vs Phase 2 report comparison, use the comparison workflow on comparable `QualityReport` objects. The cleanest current path is to compare the Phase 1 report against one company-level Phase 2 report:

```python
from synthetic_enterprise.evaluation.comparison import PhaseMetricsComparator
from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator

phase2_reports = DatasetQualityEvaluator(
    enterprise=phase2.enterprise,
    context=context,
).evaluate_by_company(phase2.rows_by_source)

phase2_company_report = next(iter(phase2_reports.values()))
comparison = PhaseMetricsComparator().compare(
    phase1_report=baseline_report,
    phase2_report=phase2_company_report,
)

print(comparison.to_dict())
```

This comparison is where duplicate reduction, lexical diversity, scenario coverage, noise entropy, and thread-depth improvement checks are enforced.

## Before Vs After

### Before

```text
Subject: Follow up
Body: follow up after review
```

Properties:

- short and repetitive
- weak style separation
- little persona or company voice
- shallow scenario context

### After

```text
Subject: FW: Review scope and owner check for Davis-Andrews
Body: Enterprise cadence: forwarding the internal validation chain before we resend the attendee recap. Procurement is still holding the owner line, so please confirm who will send the post-meeting note by Friday 14:00 UTC.
```

Properties:

- profile-specific marker
- clearer thread state
- compositional clause ordering
- richer follow-up context
- still grounded in the same event/account state

## Hard Negatives

Phase 2 hard negatives are closer to positives but remain non-relevant.

Examples:

- event mention without attendance evidence
- budget mention in an administrative context rather than buying intent
- travel complaint that looks like a complaint but is not product pain
- “follow up” note with no owner or required action
- executive visibility mention without real decision-maker engagement

Representative example:

```text
Subject: VP visibility on quarterly review follow up
Body: Adding the VP for visibility only. No sponsor approval, owner handoff, or customer action is needed in this thread.
Label: non-relevant hard negative
```

## Temporal Scenario Arcs

Representative Phase 2 arc:

```text
event_invite_to_attendance
  identified -> invite sent
  active -> accepted / tentative
  active -> reminder + internal coordination
  completed -> attended / no_show
  completed -> post-event follow-up
```

Representative escalation arc:

```text
support_escalation
  identified -> issue opened
  active -> severity raised
  stalled -> customer still blocked
  reopened -> retest fails and issue returns
```

These arcs are the basis for more realistic thread reopening, lagged CRM updates, and channel-specific follow-up behavior.

## Company Style Differentiation

The same event can render differently by company profile.

### Enterprise B2B Formal

```text
Subject: Review scope confirmation for Davis-Andrews
Body: Enterprise cadence: please confirm the final owner list before the customer recap is circulated.
```

### Fast Moving Startup

```text
Subject: quick owner check for Davis-Andrews
Body: Startup sync: need the final owner list before we fire off the recap.
```

### Security Sensitive Enterprise

```text
Subject: Access-controlled recap workflow for Davis-Andrews
Body: Security review: confirm the approved owner list before the customer recap leaves the controlled distribution.
```

The linked account and event facts can stay the same while the lexical fingerprint changes.
