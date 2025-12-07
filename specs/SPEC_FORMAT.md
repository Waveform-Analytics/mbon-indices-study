# Spec Document Format

This document defines the canonical format for all specification documents in this project.

## General Principles

1. **Use proper markdown headers** - All sections use `##` level headers
2. **One title** - The `#` header is the document title; no separate "Title" field
3. **Consistent structure** - All specs of the same type follow the same section order
4. **No decorative dividers** - Headers provide sufficient structure; avoid `---` between sections

## Stage Spec Format

```markdown
# [NN] [Name] — Stage Spec

## Purpose
- Brief statement of what this stage accomplishes

## Inputs
- `path/to/input.file` — description
- Config references: `config/analysis.yml -> key`

## Outputs
- `path/to/output.file` — description

## Methods
- Description of algorithms, transforms, or approaches
- Can include subsections with `###` if needed for complex stages

## Parameters
- `param_name`: description, default value, config reference

## Acceptance Criteria
- Checkable criteria that indicate successful completion

## Edge Cases
- Known edge cases and how they are handled

## Performance
- Runtime targets, memory considerations

## Dependencies
- Upstream: what this stage requires
- Downstream: what depends on this stage

## Change Record
- YYYY-MM-DD: Description of change
```

## ADR (Architecture Decision Record) Format

```markdown
# ADR-[NN]: [Decision Title]

## Context
- Background and problem statement
- Why a decision is needed

## Decision
- What was decided
- Key details of the approach

## Consequences
- Positive and negative outcomes
- Trade-offs accepted

## Alternatives
- Other options considered
- Why they were not chosen

## References
- Related specs, external docs, or prior art
```

## Notes

- **Subsections**: Use `###` for subsections within a `##` section when needed
- **Code blocks**: Use fenced code blocks for formulas, file paths, or code
- **Tables**: Use markdown tables for structured comparisons
- **Lists**: Use `-` for unordered lists; indent nested items with 2 spaces