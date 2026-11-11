from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def convert(files: list[Path], output: Path):
    if not files:
        raise ValueError("No DICOM image files were supplied.")
    headers = [pydicom.dcmread(str(p), stop_before_pixels=True, force=False) for p in files]
    modalities = {getattr(h, "Modality", "") for h in headers}
    if len(modalities) != 1 or not modalities.issubset({"US", "CT"}):
        raise ValueError(f"Selected series must be a single CT or ultrasound modality; found {sorted(modalities)}.")
    frames = max(int(getattr(h, "NumberOfFrames", 1)) for h in headers)
    orientations = [tuple(float(x) for x in h.ImageOrientationPatient)
                    for h in headers if hasattr(h, "ImageOrientationPatient")]
    if len(files) > 1:
        if len(orientations) != len(headers) or any(not np.allclose(orientations[0], x, rtol=0, atol=1e-5) for x in orientations[1:]):
            raise ValueError("The multi-file series lacks a consistent ImageOrientationPatient.")
        row, col = np.asarray(orientations[0][:3]), np.asarray(orientations[0][3:])
        normal = np.cross(row, col)
        projections = [float(np.dot(np.asarray(h.ImagePositionPatient, dtype=float), normal))
                       for h in headers if hasattr(h, "ImagePositionPatient")]
        if len(projections) != len(headers):
            raise ValueError("The multi-file series lacks ImagePositionPatient; spatial depth cannot be verified.")
        sorted_proj = np.sort(np.asarray(projections))
        gaps = np.diff(sorted_proj)
        if len(np.unique(np.round(sorted_proj, 5))) < 3 or np.any(gaps <= 1e-5):
            raise ValueError("At least three distinct slice positions are required to establish spatial 3D.")
        if not np.allclose(gaps, np.median(gaps), rtol=0.05, atol=0.05):
            raise ValueError("Slice positions are not regularly spaced; refusing to infer missing slice geometry.")
        order = np.argsort(projections)
        files = [files[int(i)] for i in order]
    distinct_positions = {tuple(float(x) for x in h.ImagePositionPatient)
                          for h in headers if hasattr(h, "ImagePositionPatient")}
    if len(files) < 2 and frames < 2:
        raise ValueError("Only a single 2D image is available; spatial 3D is not established.")
    if len(files) == 1:
        ds = headers[0]
        groups = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
        positions = []
        for group in groups:
            for name in ("PlanePositionSequence", "PlanePositionVolumeSequence", "PlanePositionPatientSequence"):
                seq = getattr(group, name, None)
                if seq and hasattr(seq[0], "ImagePositionPatient"):
                    positions.append(tuple(float(v) for v in seq[0].ImagePositionPatient))
                    break
        if len(positions) != frames or len(set(positions)) < 3:
            raise ValueError("Multi-frame image does not provide at least three distinct per-frame spatial positions; it may be cine.")
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(files[0]))
        reader.SetImageIO("GDCMImageIO")
        image = reader.Execute()
    else:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(p) for p in files])
        image = reader.Execute()
    if image.GetDimension() != 3 or image.GetSize()[2] < 2:
        raise ValueError(f"Decoded image is {image.GetDimension()}D with size {image.GetSize()}; spatial 3D required.")
    if any(not np.isfinite(s) or s <= 0 for s in image.GetSpacing()):
        raise ValueError(f"Invalid physical spacing: {image.GetSpacing()}")
    pixel_spacings = [getattr(h, "PixelSpacing", None) for h in headers]
    pixel_spacing_valid = all(v is not None and len(v) == 2 and all(np.isfinite(float(x)) and float(x) > 0 for x in v)
                              for v in pixel_spacings)
    image.SetMetaData("radiomics.spacing_calibrated", "true" if pixel_spacing_valid else "false")
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output), True)
    reread = sitk.ReadImage(str(output))
    if reread.GetSize() != image.GetSize() or reread.GetSpacing() != image.GetSpacing() or reread.GetOrigin() != image.GetOrigin() or reread.GetDirection() != image.GetDirection():
        output.unlink(missing_ok=True)
        raise ValueError("NRRD round-trip changed image geometry; output removed.")
    report = {"path": str(output), "dimension": image.GetDimension(), "size": list(image.GetSize()),
              "spacing": list(image.GetSpacing()), "origin": list(image.GetOrigin()),
              "direction": list(image.GetDirection()), "dtype": str(sitk.GetArrayViewFromImage(image).dtype),
              "source_file_count": len(files), "spatial_positions_detected": len(distinct_positions)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return image, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/images/volume.nrrd"))
    args = parser.parse_args()
    files = sorted((p for p in args.input_dir.rglob("*") if p.is_file()), key=lambda p: str(p).lower()) if args.input_dir else args.files
    _, report = convert(files, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
