from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pydicom

from inspect_dicomdir import _spatial_status


def inspect_archive(archive: Path):
    try:
        zf = ZipFile(archive)
    except (OSError, BadZipFile) as exc:
        raise ValueError(f"Cannot open archive: {exc}") from exc
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        dicomdir_names = [m.filename for m in members if Path(m.filename).name.upper() == "DICOMDIR"]
        referenced_names = []
        for name in dicomdir_names:
            try:
                with zf.open(name) as stream:
                    ds = pydicom.dcmread(stream, force=True, stop_before_pixels=True)
                for record in getattr(ds, "DirectoryRecordSequence", []):
                    raw = getattr(record, "ReferencedFileID", None)
                    if raw:
                        parts = list(raw) if not isinstance(raw, str) else raw.replace("\\", "/").split("/")
                        referenced_names.append("/".join(str(p) for p in parts))
            except Exception:
                continue
        names_by_lower = {m.filename.replace("\\", "/").lower(): m.filename for m in members}
        refs_resolved = 0
        for name in referenced_names:
            candidate = name.replace("\\", "/").lower()
            if candidate in names_by_lower:
                refs_resolved += 1
            else:
                suffix = "/" + candidate
                if any(existing.endswith(suffix) for existing in names_by_lower):
                    refs_resolved += 1

        inspect_members = members
        if referenced_names:
            matched = []
            for name in referenced_names:
                candidate = name.replace("\\", "/").lower()
                resolved = names_by_lower.get(candidate)
                if not resolved:
                    suffix = "/" + candidate
                    resolved = next((orig for key, orig in names_by_lower.items() if key.endswith(suffix)), None)
                if resolved:
                    matched.append(resolved)
            if matched:
                inspect_members = [next(m for m in members if m.filename == n) for n in dict.fromkeys(matched)]

        grouped = defaultdict(list)
        dicom_count = 0
        failures = 0
        for member in inspect_members:
            try:
                with zf.open(member.filename) as stream:
                    ds = pydicom.dcmread(stream, force=False, stop_before_pixels=True)
            except Exception:
                failures += 1
                continue
            if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
                continue
            dicom_count += 1
            keys = ("Modality", "StudyInstanceUID", "SeriesInstanceUID", "SOPClassUID", "SeriesDescription",
                    "NumberOfFrames", "Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
                    "ImagePositionPatient", "ImageOrientationPatient", "ImageType", "PhotometricInterpretation",
                    "PixelRepresentation", "RescaleSlope", "RescaleIntercept")
            info = {k: [str(v) for v in getattr(ds, k)] if k in ("PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient", "ImageType") and hasattr(ds, k) else str(getattr(ds, k))
                    for k in keys if hasattr(ds, k)}
            info["path_in_archive"] = member.filename
            info["per_frame_positions"] = []
            for frame in getattr(ds, "PerFrameFunctionalGroupsSequence", []):
                for seq_name in ("PlanePositionSequence", "PlanePositionVolumeSequence", "PlanePositionPatientSequence"):
                    seq = getattr(frame, seq_name, None)
                    if seq and hasattr(seq[0], "ImagePositionPatient"):
                        info["per_frame_positions"].append([float(v) for v in seq[0].ImagePositionPatient])
                        break
            grouped[str(getattr(ds, "SeriesInstanceUID", "unknown"))].append(info)

        series = []
        for uid, items in grouped.items():
            status, reason = _spatial_status(items)
            first = items[0]
            positions = [x.get("ImagePositionPatient") for x in items if x.get("ImagePositionPatient")]
            orientation = first.get("ImageOrientationPatient")
            orientation_consistent = bool(orientation) and all(
                x.get("ImageOrientationPatient") and np.allclose(
                    np.asarray(x["ImageOrientationPatient"], dtype=float),
                    np.asarray(orientation, dtype=float), rtol=0, atol=1e-5)
                for x in items)
            median_spacing = None
            spacing_cv = None
            if orientation_consistent and len(positions) >= 3:
                row = np.asarray(orientation[:3], dtype=float)
                col = np.asarray(orientation[3:], dtype=float)
                normal = np.cross(row, col)
                projections = np.sort(np.asarray(positions, dtype=float) @ normal)
                gaps = np.diff(projections)
                if len(gaps) and np.all(gaps > 0):
                    median_spacing = float(np.median(gaps))
                    spacing_cv = float(np.std(gaps) / median_spacing) if median_spacing else None
            series.append({"series_uid": uid, "modality": first.get("Modality"),
                           "study_uid": first.get("StudyInstanceUID"),
                           "description": first.get("SeriesDescription"), "file_count": len(items),
                           "rows": first.get("Rows"), "columns": first.get("Columns"),
                           "max_number_of_frames": max([int(x.get("NumberOfFrames", "1")) for x in items] or [1]),
                           "uncompressed_bytes": sum(next(m.file_size for m in members if m.filename == x["path_in_archive"]) for x in items),
                           "distinct_slice_positions": len({tuple(x) for x in positions}),
                           "pixel_spacing": first.get("PixelSpacing"), "slice_thickness": first.get("SliceThickness"),
                           "orientation_consistent": orientation_consistent,
                           "median_slice_spacing_mm": median_spacing, "slice_spacing_cv": spacing_cv,
                           "spatial_status": status, "spatial_reason": reason, "files": items})
        return {"archive": str(archive), "archive_file_count": len(members),
                "dicomdir_count": len(dicomdir_names), "referenced_file_count": len(referenced_names),
                "resolved_reference_count": refs_resolved, "inspected_file_count": len(inspect_members),
                "dicom_image_count": dicom_count, "unreadable_file_count": failures,
                "series": series, "patient_identifiers_omitted": True}


def main():
    parser = argparse.ArgumentParser(description="Inspect DICOM headers inside a ZIP without extracting or modifying it.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/inspection/archive_inspection.json"))
    args = parser.parse_args()
    report = inspect_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: report[k] for k in ("archive_file_count", "dicomdir_count", "referenced_file_count", "resolved_reference_count", "inspected_file_count", "dicom_image_count", "unreadable_file_count")}
    summary["series"] = [{k: s[k] for k in ("series_uid", "study_uid", "modality", "description", "file_count", "uncompressed_bytes", "rows", "columns", "max_number_of_frames", "distinct_slice_positions", "pixel_spacing", "slice_thickness", "orientation_consistent", "median_slice_spacing_mm", "slice_spacing_cv", "spatial_status", "spatial_reason")} for s in report["series"]]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
