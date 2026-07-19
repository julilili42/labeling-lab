# Handoff

The supported training path is:

```bash
uv run train-models run
```

Inputs and prompt versions are frozen in `training.toml`. The complete release,
including content hashes and figure-ready benchmark tables, is written to
`data/release/`. Older model files under `data/models/` are historical and are
not inputs to the new pipeline.

Teacher labels are weak supervision. Before making claims about production
quality, create a fresh human-reviewed test set and register it as a new frozen
release instead of tuning against the current benchmark.

PageVerdict combines the frozen page-v4 dataset with human-reviewed hard
negatives that are restricted to train. LinkVerdict uses only binary human
reviews. Treat both as reproducible baselines until a fresh, human-reviewed,
host-disjoint test release supports stronger production claims.
