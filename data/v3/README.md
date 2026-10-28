# Product V3 data

Use the `split_path` in `config/v3/dataset.json`:
`data/v3/frozen_readable_family_split.csv` (1,641 families).

`frozen_family_split.csv` is the preserved, unapproved pre-review draft. It is
not a training input. Barcode and redaction families found by visual inspection
were excluded in the separate reviewed split before any product training.

The font catalog, technical audit, and explicit exclusion reasons are tracked.
Downloaded fonts/licences, generated TAR shards, per-image metadata, SQLite
manifest, and caches are local ignored files, not GitHub dataset uploads.

See `docs/product/v3_data.md` for reproduction, provenance, storage, quality,
and training boundaries. A generation progress count is not a passed data gate.
