"""Generate mutation case files from bucket definitions."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from evals import eval_root, repo_root
from logging_utils import configure_logging, get_logger

DEFAULT_BUCKETS_PATH = eval_root() / "mutation_tests" / "buckets.json"
DEFAULT_OUTPUT_DIR = repo_root() / "artifacts" / "mutation_tests" / "generated_cases"
DEFAULT_PDF_OUTPUT_DIR = repo_root() / "artifacts" / "mutation_tests" / "generated_pdfs"
MUTATION_CASE_COUNT = 2
MIN_CASE_ID = 1
MAX_CASE_ID = 99
MIN_REQUESTED_COMPANY_COUNT = 5
MAX_REQUESTED_COMPANY_COUNT = 50
logger = get_logger(__name__)


def generate_mutation_case_files(
    *,
    count: int = MUTATION_CASE_COUNT,
    buckets_path: Path | None = None,
    output_dir: Path | None = None,
    pdf_output_dir: Path | None = None,
    rng: random.Random | None = None,
) -> list[Path]:
    """Generate mutation case JSON files and return their paths."""

    configure_logging()
    if count < 1:
        raise ValueError("count must be at least 1")
    if count > MAX_CASE_ID:
        raise ValueError(f"count must be less than or equal to {MAX_CASE_ID}")

    randomizer = rng or random.SystemRandom()
    bucket_file = buckets_path or DEFAULT_BUCKETS_PATH
    case_dir = output_dir or DEFAULT_OUTPUT_DIR
    pdf_dir = pdf_output_dir or DEFAULT_PDF_OUTPUT_DIR
    logger.info(
        "Generating mutation case files count=%s buckets_path=%s "
        "output_dir=%s pdf_output_dir=%s",
        count,
        bucket_file,
        case_dir,
        pdf_dir,
    )
    buckets = json.loads(bucket_file.read_text(encoding="utf-8"))
    if count > len(buckets["profession"]):
        raise ValueError("count cannot exceed the number of available professions")

    case_ids = randomizer.sample(range(MIN_CASE_ID, MAX_CASE_ID + 1), count)
    professions = randomizer.sample(buckets["profession"], count)

    case_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    for case_id, profession in zip(case_ids, professions, strict=True):
        payload = build_random_mutation_case_payload(
            buckets,
            randomizer,
            case_id=case_id,
            profession=profession,
        )
        pdf_path = _unique_pdf_path(pdf_dir, payload)
        payload["pdf_path"] = _repo_relative_path(pdf_path)
        _write_fictional_cv_pdf(pdf_path, payload, randomizer)

        case_path = _unique_case_path(case_dir, payload)
        case_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        generated_paths.append(case_path)
        logger.info(
            "Generated mutation case case_id=%s case_path=%s pdf_path=%s",
            payload["id"],
            case_path,
            pdf_path,
        )

    logger.info(
        "Generated mutation case files count=%s output_dir=%s",
        len(generated_paths),
        case_dir,
    )
    return generated_paths


def build_random_mutation_case_payload(
    buckets: dict[str, Any],
    rng: random.Random,
    *,
    case_id: int,
    profession: str | None = None,
) -> dict[str, Any]:
    """Build one mutation case payload matching the non-deterministic case schema."""

    if not MIN_CASE_ID <= case_id <= MAX_CASE_ID:
        raise ValueError(f"case_id must be between {MIN_CASE_ID} and {MAX_CASE_ID}")

    profession = profession or _choose_one(buckets["profession"], rng)
    experience = _choose_one(buckets["experience"], rng)
    goal = _choose_one(buckets["goal"], rng)
    region = _choose_one(buckets["region"], rng)
    filter_criteria = _sample_many(buckets["filter_criteria"], rng, min_count=2, max_count=4)
    axes = _sample_many(buckets["axes"], rng, min_count=2, max_count=3)
    style_description = _choose_one(buckets["communication_style"]["description"], rng)
    behavioral_traits = _sample_many(
        buckets["communication_style"]["behavioral_traits"],
        rng,
        min_count=3,
        max_count=5,
    )
    requested_company_count = rng.randint(
        MIN_REQUESTED_COMPANY_COUNT,
        MAX_REQUESTED_COMPANY_COUNT,
    )

    return {
        "id": case_id,
        "name": _slug(profession),
        "profession": profession,
        "experience": experience,
        "goal": goal,
        "first_prompt": (
            f"Could you please find {requested_company_count} companies in {region} "
            "matching my CV?"
        ),
        "filter_criteria": filter_criteria,
        "axes": axes,
        "communication_style": {
            "description": style_description,
            "behavioral_traits": behavioral_traits,
        },
    }


def _choose_one(values: list[str], rng: random.Random) -> str:
    if not values:
        raise ValueError("Cannot choose from an empty bucket")
    return rng.choice(values)


def _sample_many(
    values: list[str],
    rng: random.Random,
    *,
    min_count: int,
    max_count: int,
) -> list[str]:
    if len(values) < min_count:
        raise ValueError(f"Bucket must contain at least {min_count} values")
    count = rng.randint(min_count, min(max_count, len(values)))
    return rng.sample(values, count)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _unique_case_path(case_dir: Path, payload: dict[str, Any]) -> Path:
    base_path = case_dir / f"{payload['id']:02d}_{payload['name']}.json"
    if not base_path.exists():
        return base_path

    suffix = 2
    while True:
        case_path = case_dir / f"{payload['id']:02d}_{payload['name']}_{suffix}.json"
        if not case_path.exists():
            return case_path
        suffix += 1


def _unique_pdf_path(pdf_dir: Path, payload: dict[str, Any]) -> Path:
    base_path = pdf_dir / f"{payload['id']:02d}_{payload['name']}_cv.pdf"
    if not base_path.exists():
        return base_path

    suffix = 2
    while True:
        pdf_path = pdf_dir / f"{payload['id']:02d}_{payload['name']}_cv_{suffix}.pdf"
        if not pdf_path.exists():
            return pdf_path
        suffix += 1


def _repo_relative_path(path: Path) -> str:
    repo_root = eval_root().parent
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _write_fictional_cv_pdf(path: Path, payload: dict[str, Any], rng: random.Random) -> None:
    lines = _fictional_cv_lines(payload, rng)
    stream = _pdf_text_stream(lines)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, pdf_object in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{object_number} 0 obj\n".encode("ascii"))
        content.extend(pdf_object)
        content.extend(b"\nendobj\n")

    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(content))


def _fictional_cv_lines(payload: dict[str, Any], rng: random.Random) -> list[str]:
    case_id = payload["id"]
    profession = payload["profession"]
    first_name = _choose_one(
        [
            "Alex",
            "Maya",
            "Jordan",
            "Nina",
            "Owen",
            "Leah",
            "Samir",
            "Clara",
            "Adrian",
            "Elena",
        ],
        rng,
    )
    last_name = _choose_one(
        [
            "Morgan",
            "Patel",
            "Rivera",
            "Chen",
            "Kowalski",
            "Haddad",
            "Costa",
            "Okafor",
            "Novak",
            "Bennett",
        ],
        rng,
    )
    name = f"{first_name} {last_name}"
    companies = rng.sample(
        [
            "Northbridge Services",
            "Riverstone Group",
            "Cedarfield Partners",
            "Brightpath Labs",
            "Horizon Civic Network",
            "AtlasWorks Ltd.",
        ],
        3,
    )
    university = _choose_one(
        [
            "University of Lisbon",
            "University of Toronto",
            "University of Cape Town",
            "University of Warsaw",
            "University of Melbourne",
            "University of Edinburgh",
            "National University of Singapore",
            "University of Buenos Aires",
        ],
        rng,
    )
    degree = _choose_one(
        [
            f"Bachelor of Arts in {profession}",
            f"Bachelor of Science in {profession}",
            f"Master of Professional Studies in {profession}",
            f"Master of Science in {profession}",
        ],
        rng,
    )
    end_year = rng.randint(2018, 2025)
    job_history = [
        (f"Senior {profession}", companies[0], end_year - 7, end_year),
        (profession, companies[1], end_year - 11, end_year - 7),
        (f"Junior {profession}", companies[2], end_year - 14, end_year - 11),
    ]

    lines = [
        "Curriculum Vitae",
        f"Name: {name}",
        f"Profession: {profession}",
        f"Email: {first_name.lower()}.{last_name.lower()}{case_id}@example.com",
        f"Phone: +1-555-010-{case_id:02d}",
        "",
        "Experience",
    ]
    for title, company, start_year, finish_year in job_history:
        lines.append(f"{start_year}-{finish_year}: {title}, {company}")

    lines.extend(
        [
            "",
            "Education",
            f"{end_year - 18}-{end_year - 14}: {degree}, {university}",
            "Grade: 3.7/4.0",
        ]
    )
    return lines


def _pdf_text_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 11 Tf", "72 740 Td", "14 TL"]
    for line in lines:
        commands.append(f"({_escape_pdf_text(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
