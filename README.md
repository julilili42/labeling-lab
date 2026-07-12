# Labeling Lab

Local workbench for collecting candidates, manual and teacher-assisted labels,
and training PageVerdict and LinkVerdict releases.

```bash
uv run labeling-lab
```

Open <http://127.0.0.1:8010>. The dashboard starts the existing SERP, fetch,
auto-label, and training steps and keeps their output visible. The lower UI
labels page and link candidates manually.

`search-engine` consumes only the tested `.joblib` releases copied from
`data/models/`; it contains no labeling UI or training commands.
