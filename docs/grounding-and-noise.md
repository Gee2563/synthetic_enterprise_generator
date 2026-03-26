# Grounding And Noise

## Taxonomy

Relevant categories:

- `pain_point`
- `complaint`
- `feature_request`
- `buying_signal`
- `churn_risk`
- `event_attendance`
- `follow_up`
- `blocker`
- `escalation`
- `decision_maker_signal`

Noise categories:

- `greetings`
- `status_updates`
- `social_chatter`
- `scheduling_only`
- `fyi_forward`
- `automated_notification`
- `duplicate_summary`
- `low_signal_checkin`
- `irrelevant_marketing`
- `admin_ops`

Each generated row must have exactly one primary category.

## Grounding Rules

Relevant rows are not allowed to be free-floating.

Current grounding rules include:

- `pain_point` should map back to a customer issue or problem state
- `buying_signal` should map back to opportunity, account, or contact state
- `event_attendance` should map back to attendance-bearing event or campaign state
- `blocker` should map back to a ticket, project constraint, or opportunity constraint

The row-level grounding fields are:

- `primary_category`
- `is_relevant`
- `relevance_reason`
- `provenance`

`provenance` contains:

- `object_type`
- `object_id`
- `strength`
- `explanation`

## Provenance Strength

The project uses provenance strength to separate true positives from harder negatives.

- `strong`: relevant rows with a defensible underlying business state
- `weak`: non-relevant hard negatives that mention a business object but do not meet the relevance bar
- `null`: generic noise with no useful business linkage

## Noise Generation

Noise is generated from several layers.

### Generic Noise

Examples:

- greetings in Slack channels
- internal Teams coordination that only moves documents around
- email calendar shuffles without business content
- low-value CRM notes and stale records

### Hard Negatives

Hard negatives deliberately resemble relevant rows.

Examples:

- "let's discuss next week" around a customer event but with no attendance or follow-up obligation
- a weather or travel complaint that contains complaint-like language but is not a product issue
- a CRM task that mentions budget or timing in a non-buying or non-attendance context

Hard negatives are labeled non-relevant even though they keep lexical overlap with positives.

## Current Controls

Key knobs in `GeneratorConfig`:

- `noise_ratio`
- `hard_negative_ratio`
- `cross_system_ratio`
- `verbosity_ratio`
- `date_range`
- `company_size`

These affect overall record mix, amount of noise, and how often facts appear across multiple systems.

## Practical Interpretation

When you inspect a row:

1. Read `primary_category`.
2. Check `is_relevant`.
3. Read `relevance_reason`.
4. Inspect `provenance`.
5. Verify linked ids such as `event_id`, `ticket_id`, `opportunity_id`, or `campaign_id`.

If the row is relevant, the provenance should tell you exactly why. If the row is noisy, it should either have no provenance or only weak provenance.
