# ADR-01: Python Package Architecture for Reusable Transforms

## Context
- Prior work mixed pipeline scripts and a custom Python package for cleaning/alignment.
- We need testable, reusable transforms while allowing flexible pipelines and stage evolution.

## Decision
- Create a small Python package at `src/python/mbon_indices/` containing loaders, cleaning/alignment utilities, feature creation, and validators.
- Keep orchestration in scripts/workflows that import the package; scripts remain thin wrappers.

## Consequences
- Improves testability and reuse; decouples stage orchestration from transform implementations.
- Clear spec traceability: functions map to spec sections and acceptance criteria.
- Slight upfront effort to refactor prior scripts into package modules.

## Alternatives
- Keep all logic in scripts only (simpler initially, harder to test and reuse).
- Use a larger monolithic package (more complex, less flexible for rapid spec changes).

## References
- Stage specs in `specs/stages/00-04` referencing package functions for deterministic transforms.