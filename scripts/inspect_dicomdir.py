from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


def _text(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return [str(v) for v in value]
    return str(value)


def _ref_paths(ds, root: Path):
    refs = []
    for record in getattr(ds, "DirectoryRecordSequence", []):
        raw = getattr(record, "ReferencedFileID", None)
        if raw:
            parts = list(raw) if not isinstance(raw, str) else raw.replace("\\", "/").split("/")
            refs.append(root.joinpath(*[str(p) for p in parts]))
    return list(dict.fromkeys(refs))


def _header(path: Path):
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except (InvalidDicomError, OSError, PermissionError):
        return None
    keep = ("Modality", "StudyInstanceUID", "SeriesInstanceUID", "SOPClassUID",
            "SeriesDescription", "NumberOfFrames", "Rows", "Columns", "PixelSpacing",
            "SliceThickness", "SpacingBetweenSlices", "ImagePositionPatient",
            "ImageOrientationPatient", "ImageType", "PhotometricInterpretation",
            "PixelRepresentation", "RescaleSlope", "RescaleIntercept")
    info = {key: _text(getattr(ds, key, None)) for key in keep if hasattr(ds, key)}
    info["path"] = str(path)
    info["TransferSyntaxUID"] = _text(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None))
    info["has_shared_functional_groups"] = bool(getattr(ds, "SharedFunctionalGroupsSequence", None))
    info["has_per_frame_functional_groups"] = bool(getattr(ds, "PerFrameFunctionalGroupsSequence", None))
    frame_positions = []
    for frame in getattr(ds, "PerFrameFunctionalGroupsSequence", []):
        for name in ("PlanePositionSequence", "PlanePositionVolumeSequence", "PlanePositionPatientSequence"):
            pos = getattr(frame, name, None)
            if pos and hasattr(pos[0], "ImagePositionPatient"):
                frame_positions.append([float(v) for v in pos[0].ImagePositionPatient])
                break
    info["per_frame_position_count"] = len(frame_positions)
    info["per_frame_positions"] = frame_positions
    regions = []
    for region in getattr(ds, "SequenceOfUltrasoundRegions", []):
        regions.append({k: _text(getattr(region, k, None)) for k in (
            "RegionSpatialFormat", "RegionDataType", "RegionFlags", "PhysicalUnitsXDirection",
            "PhysicalUnitsYDirection", "PhysicalDeltaX", "PhysicalDeltaY",
            "RegionLocationMinX0", "RegionLocationMaxX1", "RegionLocationMinY0", "RegionLocationMaxY1")
                        if hasattr(region, k)})
    info["ultrasound_regions"] = regions
    return info


def _spatial_status(items):
    def regular_positions(positions, orientation):
        if len(positions) < 3 or not orientation or len(orientation) != 6:
            return False
        arr = np.asarray(positions, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return False
        row, col = np.asarray(orientation[:3], dtype=float), np.asarray(orientation[3:], dtype=float)
        normal = np.cross(row, col)
        gaps = np.diff(np.sort(arr @ normal))
        return bool(np.all(gaps > 1e-4) and np.allclose(gaps, np.median(gaps), rtol=0.05, atol=0.05))

    multi = [it for it in items if int(it.get("NumberOfFrames", 1)) > 1]
    if multi:
        positioned = [it for it in multi if regular_positions(it.get("per_frame_positions", []), it.get("ImageOrientationPatient"))]
        if positioned:
            return "true_spatial_3d_candidate", "Per-frame plane positions are present; verify distinct positions and orientations during conversion."
        return "cine_or_unverified_multiframe", "Multiple frames exist, but distinct spatial frame positions are not established."
    positions = [it.get("ImagePositionPatient") for it in items if it.get("ImagePositionPatient")]
    orientation = items[0].get("ImageOrientationPatient") if items else None
    if regular_positions(positions, orientation):
        return "spatially_ordered_slices_candidate", "At least three slice positions have a consistent orientation and regular physical spacing."
    if len(positions) >= 3:
        return "spatial_geometry_unverified", "Distinct positions exist, but orientation or regular slice spacing is missing/irregular."
    if items and items[0].get("Rows") and items[0].get("Columns"):
        return "single_2d_or_insufficient_geometry", "No verifiable spatial depth is established from the available image positions."
    return "unknown", "Insufficient metadata to establish spatial geometry."


def inspect(dicomdir: Path, output: Path, fallback_scan: bool = True):
    root = dicomdir.parent
    index_exists = dicomdir.is_file()
    refs = _ref_paths(pydicom.dcmread(str(dicomdir), force=True, stop_before_pixels=True), root) if index_exists else []
    resolved = [p for p in refs if p.is_file()]
    missing = [str(p) for p in refs if not p.is_file()]
    discovered = list(resolved)
    scan_count = 0
    if fallback_scan and (missing or not index_exists):
        seen = {p.resolve() for p in discovered}
        for p in root.rglob("*"):
            if not p.is_file() or p.resolve() in seen or p == dicomdir:
                continue
            scan_count += 1
            discovered.append(p)
    files = []
    for path in discovered:
        info = _header(path)
        if info:
            files.append(info)
    series = defaultdict(list)
    for item in files:
        key = item.get("SeriesInstanceUID", "unknown")
        series[str(key)].append(item)
    summaries = []
    for uid, items in series.items():
        status, reason = _spatial_status(items)
        first = items[0]
        summaries.append({
            "series_uid": uid, "modality": first.get("Modality"),
            "description": first.get("SeriesDescription"), "file_count": len(items),
            "rows": first.get("Rows"), "columns": first.get("Columns"),
            "number_of_frames": max([int(x.get("NumberOfFrames", 1)) for x in items] or [1]),
            "spatial_status": status, "spatial_reason": reason,
            "files": items,
        })
    report = {
        "dicomdir": str(dicomdir), "dicomdir_exists": index_exists, "referenced_file_count": len(refs),
        "resolved_reference_count": len(resolved), "missing_references": missing,
        "controlled_fallback_scan_files_examined": scan_count,
        "candidate_image_count": len(files), "series": summaries,
        "patient_identifiers_omitted": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect DICOMDIR references and image geometry without reporting patient identifiers.")
    parser.add_argument("--dicomdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/inspection/dicom_inspection.json"))
    args = parser.parse_args()
    report = inspect(args.dicomdir, args.output)
    summary = {k: report[k] for k in ("dicomdir", "dicomdir_exists", "referenced_file_count", "resolved_reference_count", "candidate_image_count", "controlled_fallback_scan_files_examined")}
    summary["missing_reference_count"] = len(report["missing_references"])
    summary["missing_reference_examples"] = report["missing_references"][:5]
    print(json.dumps(summary, indent=2))
    for s in report["series"]:
        print(f"{s['modality'] or 'Unknown'} | {s['description'] or 'Undescribed'} | files={s['file_count']} | frames={s['number_of_frames']} | {s['spatial_status']}")


if __name__ == "__main__":
    main()
