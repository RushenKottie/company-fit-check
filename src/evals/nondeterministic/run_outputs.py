"""Write output files produced by non-deterministic eval runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from evals.nondeterministic.models import TranscriptTurn
from models.artifacts import GeneratedCsvArtifact


def suite_stamp() -> str:
    """Return one UTC timestamp string for grouping case outputs."""

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def case_output_dir(
    *,
    artifact_root: Path,
    suite_stamp: str,
    case_id: int,
    run_id: str,
) -> Path:
    """Return the output directory for one case run."""

    return artifact_root / suite_stamp / str(case_id) / run_id


def write_transcript(
    *,
    artifact_root: Path,
    suite_stamp: str,
    case_id: int,
    case_name: str,
    run_id: str,
    pdf_path: str,
    turns: list[TranscriptTurn],
    status: str,
    error: str | None,
) -> str:
    """Write one transcript JSON file and return its path."""

    output_dir = case_output_dir(
        artifact_root=artifact_root,
        suite_stamp=suite_stamp,
        case_id=case_id,
        run_id=run_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "case_id": case_id,
                "case_name": case_name,
                "run_id": run_id,
                "pdf_path": pdf_path,
                "status": status,
                "error": error,
                "turns": [turn.model_dump(mode="json") for turn in turns],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(transcript_path)


def write_csv_artifact(
    *,
    artifact_root: Path,
    suite_stamp: str,
    case_id: int,
    run_id: str,
    artifact: GeneratedCsvArtifact,
) -> str:
    """Write one generated CSV artifact and return its path."""

    output_dir = case_output_dir(
        artifact_root=artifact_root,
        suite_stamp=suite_stamp,
        case_id=case_id,
        run_id=run_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / artifact.filename
    csv_path.write_bytes(artifact.content_bytes)
    return str(csv_path)


def rewrite_transcript_status(
    transcript_path: str,
    *,
    status: str,
    error: str | None,
) -> None:
    """Update status and error fields in an existing transcript file."""

    path = Path(transcript_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["error"] = error
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
