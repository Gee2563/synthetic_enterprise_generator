# Synthetic Enterprise Simulator

Synthetic Enterprise Simulator is a Python project for generating deterministic, high-noise synthetic enterprise datasets across communication and system-of-record sources.

## Goals

- Model company state first, then render source records from that state.
- Generate realistic relevant and irrelevant records for downstream retrieval and labeling tasks.
- Keep generation deterministic through explicit seeds.
- Support scalable exports for large datasets using `pandas` and `pyarrow`.
- Maintain a TDD-friendly, modular Python codebase with a `src/` layout.

## Current Status

This repository currently contains the initial scaffold only:

- package structure
- placeholder modules
- basic configuration loading
- deterministic generator context
- starter tests and CI configuration

## Quick Start

```bash
make install
make test
```

## Planned Sources

- Email
- Slack
- Microsoft Teams
- Salesforce CRM
