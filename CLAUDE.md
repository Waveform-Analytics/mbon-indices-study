Please refer to rules in .trae/rules

Please be aware of the .trae/documents folder, Especially .trae/documents/MBON Acoustic Indices Analysis — Implementation Plan.md

## Workflow Preferences

**Collaborative Learning Approach:**
- Keep prompts SHORT - one thing at a time
- User reviews EVERY file created before proceeding
- Point user to relevant files to review and WAIT for their response
- This is a learning experience as much as a building experience
- Step-by-step collaboration, not autonomous implementation

- Make sure to commit often/as needed.

## Documentation Site

The results are documented in a Quarto site at `docs/results-viewer.qmd`.

**Rebuild process:**
```bash
./rebuild-docs.sh
```

This script:
1. Copies figures from `results/figures/` to `docs/figures/`
2. Copies tables from `results/tables/` to `docs/tables/`
3. Renders `docs/results-viewer.qmd` to HTML

**When adding new figures/tables:**
1. Save outputs to `results/figures/` or `results/tables/` (source of truth)
2. Update `rebuild-docs.sh` to include the new files in the copy step
3. Add references to the new files in `docs/results-viewer.qmd`
4. Run `./rebuild-docs.sh` to rebuild

**View locally:** `open docs/results-viewer.html`
