# Knowledge

## Label contract

A positive PageVerdict destination is mainly English, substantively related to
Tuebingen or Landkreis Tuebingen, useful to general users, and not a specialist
research artifact. Institution overviews, study information, patient pages,
and public announcements can be positive. Papers, datasets, publication lists,
individual researcher profiles, booking pages, interchangeable listings, and
content-free pages are negative.

LinkVerdict predicts the same destination property before fetching. Link labels
must therefore describe the fetched target, never whether an earlier crawler
happened to enqueue or save it.

## Known data failures

- A stale-field merge overwrote 629 fresh page-v4 teacher decisions. The fixed
  canonical input is `data/page_v4_labels_clean.jsonl`.
- The link-v2 teacher marked many explicit aggregators positive. The current
  LinkVerdict release therefore uses only binary human reviews; archived
  teacher runs are not training inputs.
- Teacher labels are weak supervision. A human-reviewed, host-disjoint test set
  is still required before claiming real-world search quality.

## Resume behavior

Search, fetch, and label append JSONL rows incrementally. `--resume` requires
the same prompt version, prompt hash, teacher, and model; `--refresh` cannot be
combined with resume. Link identity is `(parent_url, target_url, anchor)`.

The default local teacher is `qwen2.5:7b` through Ollama. Prompt changes require
a new output file.

## Release discipline

`training.toml` is the single source of truth. Run `uv run train-models run` to
materialize hashed, host-disjoint splits, train both models, and create the
central benchmark. Never tune against `data/release/datasets/*/test.jsonl`.
