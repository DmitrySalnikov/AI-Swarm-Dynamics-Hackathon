"""Check the local environment without downloading models or calling providers."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required")
    for module in (
        "numpy", "scipy", "sklearn", "matplotlib", "openai", "tokenizers",
        "huggingface_hub",
    ):
        importlib.import_module(module)
    print(f"Dependencies OK; Python {sys.version.split()[0]}")

    sources = sorted((ROOT / "scripts").glob("*.py")) + sorted(
        (ROOT / "embedding_eval").glob("*.py")
    )
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"Python syntax OK: {len(sources)} files")

    all_ids: set[str] = set()
    for name in ("set_3", "set_4"):
        path = ROOT / "bench" / f"{name}.jsonl"
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(items) != 10:
            raise ValueError(f"{name}: expected 10 questions, found {len(items)}")
        for item in items:
            required = {"id", "set", "domain", "prompt", "answer", "check"}
            if not required <= item.keys() or item["set"] != name:
                raise ValueError(f"Invalid benchmark item in {path}")
            if item["id"] in all_ids or not item["prompt"]:
                raise ValueError(f"Duplicate ID or empty prompt: {item['id']}")
            if item["check"]["type"] not in {"exact", "list", "num"}:
                raise ValueError(f"Unknown check type: {item['id']}")
            all_ids.add(item["id"])
    print(f"Benchmarks OK: {len(all_ids)} unique questions")

    sys.path.insert(0, str(ROOT / "embedding_eval"))
    import numpy as np
    from dynamics import metrics_at

    aligned = metrics_at(np.tile([1.0, 0.0, 0.0], (10, 1)))
    np.testing.assert_allclose(
        [aligned["R"], aligned["spread_total"], aligned["eff_dim"]],
        [1.0, 0.0, 0.0], atol=1e-10,
    )
    orthogonal = metrics_at(np.eye(10))
    np.testing.assert_allclose(
        [orthogonal["R"], orthogonal["spread_total"], orthogonal["eff_dim"]],
        [1 / np.sqrt(10), 0.9, 9.0], atol=1e-10,
    )
    print("Metrics OK: aligned and orthogonal reference vectors")
    print("Setup ready. No provider requests were made.")


if __name__ == "__main__":
    main()
