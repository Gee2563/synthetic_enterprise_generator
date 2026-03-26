# Limitations And Ethics

## Current Limitations

### Taxonomy Coverage

The project taxonomy includes more relevant categories than the current renderers fully exercise in every run. Some labels are implemented more deeply than others.

### Realism Ceiling

The system aims for believable enterprise structure and communication patterns, but generated text will still show template boundaries and distribution artifacts.

### Benchmark Bias

Because the data is synthetic, category balance, noise ratios, and lexical patterns reflect design choices in the simulator. They do not automatically reflect the real world.

### Channel Simplification

The current sources capture useful structural differences between email, Slack, Teams, and Salesforce, but they are still simplified abstractions of those systems.

## Ethical Notes

### Do Not Present Synthetic Data As Real

Synthetic outputs should be clearly labeled as synthetic in research, benchmarking, and demos.

### Avoid Memorization-Oriented Design

The project is intended to teach models structural relevance patterns, not to imitate proprietary corpora or reuse real enterprise content.

### Human Review Still Matters

Gold sets and training exports are useful accelerators, not replacements for human validation. Before using a dataset for serious benchmarking, review:

- label quality
- rationale quality
- category coverage
- bias in noise patterns
- unintended leakage

### Privacy And Safety

Even though records are synthetic, downstream usage should still follow normal privacy and security standards for generated data pipelines, access controls, and model evaluation assets.

## Recommended Safeguards

- version and archive manifests with every dataset
- record seeds and config used for published benchmarks
- inspect sample rows before large releases
- compare quality reports across runs
- keep a reviewed gold subset for regression testing
