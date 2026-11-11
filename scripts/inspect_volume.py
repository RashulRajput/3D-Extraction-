from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def inspect(path: Path):
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayViewFromImage(image)
    return {
        "path": str(path), "dimension": image.GetDimension(), "size_xyz": list(image.GetSize()),
        "array_shape_zyx": list(arr.shape), "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()), "direction": list(image.GetDirection()),
        "dtype": str(arr.dtype), "min": float(np.min(arr)), "max": float(np.max(arr)),
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "percentiles_1_5_50_95_99": [float(x) for x in np.percentile(arr, [1, 5, 50, 95, 99])],
        "spacing_calibrated": image.HasMetaDataKey("radiomics.spacing_calibrated") and image.GetMetaData("radiomics.spacing_calibrated") == "true",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, nargs="?", default=Path("data/images/volume.nrrd"))
    parser.add_argument("--output", type=Path, default=Path("data/inspection/volume_inspection.json"))
    args = parser.parse_args()
    report = inspect(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
