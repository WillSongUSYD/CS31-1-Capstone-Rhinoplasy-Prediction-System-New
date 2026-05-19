import argparse
import csv
from collections import Counter

from .config import (
    ANNOTATION_TEMPLATE_PATH,
    CASES_TEMPLATE_PATH,
    MANIFEST_PATH,
    NOTES_TEMPLATE_PATH,
    SOURCE_DIR,
    SUMMARY_PATH,
    ensure_directories,
)
from .dataset_tools import build_manifest, write_summary


MANIFEST_FIELDS = [
    "sample_id",
    "source_kind",
    "source_container",
    "source_name",
    "source_member",
    "width",
    "height",
    "phash",
    "is_duplicate",
    "duplicate_reason",
    "duplicate_of",
    "pre_path",
    "post_path",
]


def write_manifest() -> None:
    ensure_directories()
    records = build_manifest(SOURCE_DIR)

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())

    canonical_records = [r for r in records if not r.is_duplicate]
    duplicate_counter = Counter(r.duplicate_reason or "canonical" for r in records)
    size_counter = Counter((r.width, r.height) for r in records)

    with ANNOTATION_TEMPLATE_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_id", "cost_value", "text_note", "surgery_type"])
        writer.writeheader()
        for record in canonical_records:
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "cost_value": "",
                    "text_note": "",
                    "surgery_type": "",
                }
            )

    with CASES_TEMPLATE_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_id", "cost_value", "surgery_type"])
        writer.writeheader()
        for record in canonical_records:
            writer.writerow({"sample_id": record.sample_id, "cost_value": "", "surgery_type": ""})

    with NOTES_TEMPLATE_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_id", "text_note"])
        writer.writeheader()
        for record in canonical_records:
            writer.writerow({"sample_id": record.sample_id, "text_note": ""})

    write_summary(
        SUMMARY_PATH,
        {
            "source_directory": str(SOURCE_DIR),
            "total_records": len(records),
            "canonical_records": len(canonical_records),
            "duplicate_records": len(records) - len(canonical_records),
            "duplicate_breakdown": dict(duplicate_counter),
            "top_image_sizes": [{"size": list(size), "count": count} for size, count in size_counter.most_common(10)],
            "annotation_template": str(ANNOTATION_TEMPLATE_PATH),
            "cases_template": str(CASES_TEMPLATE_PATH),
            "notes_template": str(NOTES_TEMPLATE_PATH),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the CS31 rhinoplasty dataset.")
    parser.parse_args()
    write_manifest()
    print(f"Wrote manifest to {MANIFEST_PATH}")
    print(f"Wrote dataset summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

