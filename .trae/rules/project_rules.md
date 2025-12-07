This project uses spec-driven development. All stages have associated specs that detail dependencies, inputs, outputs, and change records. The spec index is available at [specs/_index.md](specs/_index.md).

Maintain a tidy folder structure.

At the start of each session, review specs files, requirements, checklists, and establish exactly where we are in the project/pipeline, and whether there are any conflicts, inconsistencies, or gaps between specs, requirements, and checklists.

Source of Truth
- Config files (`config/*.yml`) are the source of truth for operational values (thresholds, strategies, parameters).
- Specs must reference config keys and avoid duplicating concrete values.
- When behavior changes, update config first; append spec Change Records and add ADRs for architectural decisions.

Tooling Preferences
- Python environments via uv (`pyproject.toml`, `uv.lock`); lint/format with ruff.
- Pipelines (Snakefile) are optional early; adopt for reproducible orchestration across Python/R when beneficial.

Documentation Standards
- Include concise docstrings for modules, functions, and classes to explain purpose, inputs, and outputs.
- Add short, essential comments where logic is non-obvious; prefer docstrings over excessive inline comments.
- Keep verification scripts readable with pretty-printed outputs for terminal usage.
- Spec documents follow the format defined in [specs/SPEC_FORMAT.md](specs/SPEC_FORMAT.md). Use proper markdown headers (`##`) for all sections.

Logging Standard
- Each stage script produces a timestamped log: `results/logs/stage<NN>_<name>_YYYYMMDD_HHMMSS.txt`
- Previous logs archived to `results/logs/archive/` on new runs
- Logs capture: data loading, transformations, decisions, outputs, warnings, timestamps
- Applies to all stages (Python and R scripts)

Operational Gates
- Use stage checklists for planning approval and for implementation PRs.
- PRs link specs, checklists, config changes, artifacts, and acceptance ticks.


Commit changes to specs, requirements, checklists, and project rules as soon as they are complete.