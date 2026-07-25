from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="data/sample")
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected = frame.groupby("category", group_keys=False).head(args.per_class)
    rows = []
    for _, row in selected.iterrows():
        destination = output / f"{row['category']}__{Path(row['image_path']).name}"
        with Image.open(row["image_path"]) as image:
            image.save(destination)
        rows.append({**row.to_dict(), "sample_path": str(destination)})
    pd.DataFrame(rows).to_csv(output / "sample_manifest.csv", index=False)
    print(f"Saved {len(rows)} examples to {output}")


if __name__ == "__main__":
    main()
