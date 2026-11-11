from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def same_geometry(image: sitk.Image, mask: sitk.Image, atol: float = 1e-4) -> bool:
    return (
        image.GetDimension() == mask.GetDimension() == 3
        and image.GetSize() == mask.GetSize()
        and np.allclose(image.GetSpacing(), mask.GetSpacing(), rtol=0, atol=atol)
        and np.allclose(image.GetOrigin(), mask.GetOrigin(), rtol=0, atol=atol)
        and np.allclose(image.GetDirection(), mask.GetDirection(), rtol=0, atol=atol)
    )


def import_matching_mask(archive_path: Path, image_path: Path, output_path: Path, label: int = 1):
    image = sitk.ReadImage(str(image_path))
    matched = []
    checked = []
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as temp:
        temp_dir = Path(temp)
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith((".nii", ".nii.gz", ".nrrd", ".mha")):
                continue
            candidate_path = temp_dir / Path(member.filename).name
            candidate_path.write_bytes(archive.read(member))
            candidate = sitk.ReadImage(str(candidate_path))
            compatible = same_geometry(image, candidate)
            labels, counts = np.unique(sitk.GetArrayViewFromImage(candidate), return_counts=True)
            label_count = int(np.count_nonzero(sitk.GetArrayViewFromImage(candidate) == label))
            checked.append({
                "member": member.filename,
                "size_xyz": list(candidate.GetSize()),
                "spacing_xyz": list(candidate.GetSpacing()),
                "origin_xyz": list(candidate.GetOrigin()),
                "geometry_matches_image": compatible,
                "labels": {str(int(k)): int(v) for k, v in zip(labels, counts)},
                "selected_label_voxels": label_count,
            })
            if compatible and label_count:
                matched.append((member.filename, candidate))

    if len(matched) != 1:
        raise ValueError(f"Expected exactly one non-empty mask matching the image; found {len(matched)}")

    member_name, mask = matched[0]
    # NIfTI stores coordinates as float32, which can introduce tiny header
    # roundoff. The geometry match above establishes correspondence; copy the
    # source image's exact geometry for the project's strict downstream QC.
    mask.CopyInformation(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(mask, str(output_path), useCompression=True)
    report = {
        "archive": str(archive_path),
        "image": str(image_path),
        "selected_member": member_name,
        "output_mask": str(output_path),
        "label": label,
        "candidates": checked,
    }
    report_path = output_path.with_suffix(".import.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Import the one 3D mask matching an image geometry from a mask archive.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=Path("data/images/volume.nrrd"))
    parser.add_argument("--output", type=Path, default=Path("data/masks/ROI_label.nrrd"))
    parser.add_argument("--label", type=int, default=1)
    args = parser.parse_args()
    report = import_matching_mask(args.archive, args.image, args.output, args.label)
    print(json.dumps({"selected_member": report["selected_member"], "output_mask": report["output_mask"]}, indent=2))


if __name__ == "__main__":
    main()
