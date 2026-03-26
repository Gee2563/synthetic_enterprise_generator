# Operations

## Install And Validate

```bash
make install
make test
make lint
make typecheck
```

## Run Small Generation

Generate one company snapshot only:

```bash
synthetic-enterprise simulate-company \
  --seed 17 \
  --output-root out/company \
  --account-count 4 \
  --min-employees 30 \
  --max-employees 30
```

Generate a small full dataset:

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

Generate a single source:

```bash
synthetic-enterprise generate-email --seed 17 --output-root out/email --email-count 200
synthetic-enterprise generate-slack --seed 17 --output-root out/slack --slack-count 200
synthetic-enterprise generate-teams --seed 17 --output-root out/teams --teams-count 200
synthetic-enterprise generate-crm --seed 17 --output-root out/crm --crm-count 200
```

## Important Generation Flags

- `--seed`: root deterministic seed
- `--output-root`: output directory
- `--account-count`: number of customer accounts
- `--min-employees`, `--max-employees`: company size bounds
- `--noise-ratio`: generic noise intensity
- `--hard-negative-ratio`: hard-negative intensity
- `--cross-system-ratio`: proportion of events rendered across multiple systems
- `--chunk-size`: writer chunk size
- `--write-csv`: optional CSV export in addition to Parquet

## Scale To Large Generation

Use streaming mode for large runs:

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

Why this matters:

- chunk seeds stay deterministic
- ids remain unique across chunks
- row counts aggregate cleanly
- schema stays compatible between chunked and non-chunked output
- generation does not require the full dataset to stay in memory

## Output Layout

Example output tree:

```text
out/large/
  company_state.json
  source=email/
    manifest.json
    parquet/
      part-00000.parquet
      part-00001.parquet
  source=slack/
  source=teams/
  source=salesforce/
```

## Evaluate Output Quality

Generate a quality report:

```bash
synthetic-enterprise quality-report \
  --input-root out/small \
  --output-path out/small/quality_report.json
```

The quality report summarizes:

- relevance mix
- category balance
- thread depth
- average message lengths
- cross-system linkage
- hard-negative rate
- duplicate rate
- lexical diversity
- entity coverage

## Build A Gold Set

Create a smaller benchmark set with strong rationales:

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

Exports:

- `gold_set.parquet`
- `gold_set.jsonl`

## Operational Advice

- Start with a small run and inspect manifests before large generation.
- Tune `noise_ratio` and `hard_negative_ratio` before scaling.
- Use quality reports to detect degenerate datasets early.
- Keep seeds fixed when comparing prompt or model experiments.
