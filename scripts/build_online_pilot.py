"""Build the provisional online pilot artifact from verified local COS downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pilot.prototype import extract_embeddings, fit_artifact, load_examples, write_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    examples = load_examples(args.receipts, args.images)
    embeddings = extract_embeddings(tuple(item.image_path for item in examples))
    artifact = fit_artifact(examples, embeddings)
    write_artifact(args.output, artifact)
    print(json.dumps({
        "artifact": str(args.output),
        "training_count": artifact.training_count,
        "holdout_count": artifact.holdout_count,
        "holdout_decided": artifact.holdout_decided,
        "holdout_correct": artifact.holdout_correct,
        "holdout_inconclusive": artifact.holdout_inconclusive,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
