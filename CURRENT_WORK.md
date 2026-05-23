# Current Work

## Status
No active work in progress. Submission readiness cleanup is complete.

## Completed (2026-05-04)

- Synchronized capstone/design documentation to reflect the implemented submission state:
  - `doc/capstone/project_purpose.md`
  - `doc/capstone/architecture.md`
  - `doc/capstone/final_report.md`
  - `doc/design/team_review_assignments.md`
- Created editable presentation deck:
  - `doc/capstone/spy_expiry_capstone_presentation.pptx`
- Added the survivorship-bias proxy universe artifact used by the report:
  - `cache/sp500_constituents_2015.csv`
- Fixed `DataValidator` so blocking data-quality checks align with the test suite:
  - OHLC logic violations now fail validation.
  - Frequent close-price outliers are reported.
  - `pct_change(fill_method=None)` avoids pandas forward-fill warnings.
- Verified repository tests:
  - `.venv/bin/python -m pytest tests/ -q`
  - Result: `69 passed, 4 subtests passed`
- Verified the PPTX by exporting it through LibreOffice to PDF and generating 13 preview PNGs in `/tmp/capstone_pptx_verify/`.

## Remaining
None blocking submission.
