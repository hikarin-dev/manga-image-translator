# Agent rules — Shiori translation server

Rules for ANY coding agent or AI tool working in this repository (not just Claude).

## Pipeline code and stage builds (paramount)

Clients keep each page's pipeline data (snapshots) and reuse it on a later translation, skipping
every stage whose **build** is unchanged. A build is computed automatically from the code, models,
packages and parameters behind a stage (`manga_translator/stages.py`) — nobody bumps it by hand.
It is only as complete as the lists that say what each stage runs, so:

- **Any code a stage runs must be registered in `manga_translator/stages.py`**: add a function or
  class to `CALLABLES`, or a module to `MODULES`, under the stage whose output it shapes. Code
  that runs but cannot change any output (progress, logging, telemetry, hand-off) goes in
  `NOT_OUTPUT` instead. Never hand-edit or cache a build value.
- `test/test_stage_coverage.py` traces a real gallery run and fails on any pipeline function in
  neither place. Do not silence it: register the function. When unsure which, hash it — a needless
  rerun is cheap, reusing stale output is silent and wrong.
- Shared orchestration that changes what a stage produces without touching anything listed for it
  needs a bump of `REVISIONS[stage]` (see the checklist at the top of `stages.py`).
- Restart the server after code changes: builds describe the code the process loaded at start-up.

Format and protocol: `PIPELINE_DATA.md`. Tests: `venv/Scripts/python.exe -m pytest test -q`.
