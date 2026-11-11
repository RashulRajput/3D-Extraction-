from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _geometry(image):
    return {
        "dimension": image.GetDimension(), "size": list(image.GetSize()),
        "spacing": list(image.GetSpacing()), "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def verify(image_path: Path, mask_path: Path, roi_label: int = 1):
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    a, b = sitk.GetArrayViewFromImage(image), sitk.GetArrayViewFromImage(mask)
    labels, counts = np.unique(b, return_counts=True)
    positive = b == roi_label
    bbox = None
    if np.any(positive):
        coords = np.where(positive)
        bbox = {axis: [int(np.min(v)), int(np.max(v))] for axis, v in zip(("z", "y", "x"), coords)}
    ig, mg = _geometry(image), _geometry(mask)
    checks = {
        "image_is_3d": ig["dimension"] == 3,
        "mask_is_3d": mg["dimension"] == 3,
        "size_matches": ig["size"] == mg["size"],
        "spacing_matches": np.allclose(ig["spacing"], mg["spacing"], rtol=0, atol=1e-8),
        "origin_matches": np.allclose(ig["origin"], mg["origin"], rtol=0, atol=1e-8),
        "direction_matches": np.allclose(ig["direction"], mg["direction"], rtol=0, atol=1e-8),
        "mask_non_empty": bool(np.any(b > 0)),
        "integer_labels": np.issubdtype(b.dtype, np.integer),
        "roi_label_present": bool(np.any(b == roi_label)),
    }
    spacing_valid = all(np.isfinite(s) and s > 0 for s in ig["spacing"])
    calibrated = image.HasMetaDataKey("radiomics.spacing_calibrated") and image.GetMetaData("radiomics.spacing_calibrated") == "true"
    physical_volume = int(np.count_nonzero(b == roi_label)) * float(np.prod(ig["spacing"])) if spacing_valid and calibrated else None
    return {
        "image": {"path": str(image_path), **ig},
        "mask": {"path": str(mask_path), **mg, "array_dtype": str(b.dtype)},
        "roi": {"label": roi_label, "voxel_count": int(np.count_nonzero(b == roi_label)),
                "bounding_box_zyx_inclusive": bbox, "physical_volume_mm3": physical_volume,
                "physical_spacing_calibrated": calibrated,
                "available_labels": {str(int(k)): int(v) for k, v in zip(labels, counts)}},
        "checks": checks, "geometry_check": "PASS" if all(checks.values()) else "FAIL",
        "image_array_shape_zyx": list(a.shape), "mask_array_shape_zyx": list(b.shape),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("data/images/volume.nrrd"))
    parser.add_argument("--mask", type=Path, default=Path("data/masks/ROI_label.nrrd"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.image, args.mask)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for label, section in (("IMAGE INFORMATION", report["image"]), ("MASK INFORMATION", report["mask"])):
        print(label)
        for key in ("dimension", "size", "spacing", "origin", "direction"):
            print(f"{key.title()}: {section[key]}")
    print("ROI INFORMATION")
    for key, value in report["roi"].items():
        print(f"{key}: {value}")
    print(f"GEOMETRY CHECK: {report['geometry_check']}")
    if report["geometry_check"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
