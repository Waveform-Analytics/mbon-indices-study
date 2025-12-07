# [NN] [Name] — Stage Spec

## Purpose
- Brief statement of what this stage accomplishes

## Inputs
- `path/to/input.file` — description
- Config references: `config/analysis.yml -> key`

## Outputs
- `path/to/output.file` — description
- `results/logs/stage<NN>_<name>_YYYYMMDD_HHMMSS.txt` — timestamped execution log

## Methods
- Algorithms and approaches
- Can use `###` subsections for complex stages

## Parameters
- `param_name`: description, default, config reference

## Acceptance Criteria
- Checkable criteria for successful completion

## Edge Cases
- Known edge cases and handling

## Performance
- Runtime targets
- Memory considerations

## Dependencies
- Upstream: required inputs
- Downstream: stages that depend on this

## Change Record
- YYYY-MM-DD: Description of change