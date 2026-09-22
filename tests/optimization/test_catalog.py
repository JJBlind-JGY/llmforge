import json
from pathlib import Path

from llmforge.optimization.catalog import (
    CandidateClass,
    load_candidate_catalog,
)


def test_catalog_classifies_existing_knobs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.json"

    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "watermark",
                        "title": "Watermark",
                        "area": "kv",
                        "classification": ("upstream_equivalent"),
                        "implementation": None,
                        "current_upstream_controls": ["watermark"],
                        "why_not_parameter_only": "",
                        "evidence_needed": [],
                        "notes": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    entry = load_candidate_catalog(path)[0]

    assert entry.classification == CandidateClass.UPSTREAM_EQUIVALENT
