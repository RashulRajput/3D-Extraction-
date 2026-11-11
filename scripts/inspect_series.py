from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from inspect_dicomdir import _header, _spatial_status


def inspect_files(paths):
    grouped = defaultdict(list)
    for path in paths:
        info = _header(Path(path))
        if info:
            grouped[str(info.get("SeriesInstanceUID", "unknown"))].append(info)
    series = []
    for uid, items in grouped.items():
        status, reason = _spatial_status(items)
        first = items[0]
        series.append({"series_uid": uid, "modality": first.get("Modality"),
                       "description": first.get("SeriesDescription"), "file_count": len(items),
                       "rows": first.get("Rows"), "columns": first.get("Columns"),
                       "spatial_status": status, "spatial_reason": reason, "files": items})
    return {"input_file_count": len(paths), "valid_dicom_file_count": sum(map(len, grouped.values())),
            "series": series, "patient_identifiers_omitted": True}


def main():
    parser = argparse.ArgumentParser(description="Inspect an explicit list of DICOM image files without scanning other folders.")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/inspection/series_inspection.json"))
    args = parser.parse_args()
    report = inspect_files(args.files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for series in report["series"]:
        print(f"{series['modality'] or 'Unknown'} | {series['description'] or 'Undescribed'} | {series['file_count']} files | {series['spatial_status']} | {series['spatial_reason']}")


if __name__ == "__main__":
    main()
