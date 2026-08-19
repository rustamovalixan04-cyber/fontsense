# FontSense V2 official font selection

V2 uses a pinned snapshot of the official [Google Fonts repository](https://github.com/google/fonts) at commit `e1118da94a8cb00cf6d06cdac9ef13eb1e5c6ab7`.

The discovery scan read 2,023 official `METADATA.pb` files. After requiring a supported category, a Latin subset, a compatible open licence, and a TTF/OTF file, it found 357 serif, 728 sans serif, 495 display, 251 handwriting, and 45 monospace families. The catalog retained up to 70 deterministic candidates per category; monospace retained all 45 because that is the complete eligible official set in this snapshot.

Every one of the 325 retained candidates was downloaded from a commit-pinned official URL. Its licence file was saved, and Pillow opened the font and rendered visible Latin text. All 325 passed.

The frozen V2 split uses seed 42 and exactly 40 families in each category:

- 28 training families per category;
- 6 validation families per category;
- 6 test families per category.

This gives 200 independent families: 140 train, 30 validation, and 30 test. No family crosses splits. All 15 historical V1 test families are excluded from V2, and all 30 V2 test families are new relative to the complete selected V1 family set.

Evidence:

- `data/v2/google_fonts_candidates.csv`
- `data/v2/google_fonts_audit.csv`
- `data/v2/frozen_family_split.csv`
- `reports/v2/data/candidate_discovery.json`
- `reports/v2/data/font_audit_summary.json`
- `reports/v2/data/family_split_summary.json`
