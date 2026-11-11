from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import radiomics
import SimpleITK as sitk
import yaml
from radiomics import featureextractor

try:
    from .quality_control import validate
except ImportError:
    from quality_control import validate


def extract(image_path: Path, mask_path: Path, config_path: Path, result_dir: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = dict(cfg.get("settings", {}))
    label = int(settings.pop("label", 1))
    qc = validate(image_path, mask_path, label)
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    image_array = sitk.GetArrayViewFromImage(image)
    roi = sitk.GetArrayViewFromImage(mask) == 1
    roi_values = image_array[roi]
    if roi_values.size < 8:
        raise ValueError(f"ROI has {roi_values.size} voxels; at least 8 are required for 3D texture extraction.")
    settings["force2D"] = False
    settings["resampledPixelSpacing"] = None
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    for image_type, options in cfg.get("image_types", {"Original": {}}).items():
        extractor.enableImageTypeByName(image_type, customArgs=options or {})
    extractor.disableAllFeatures()
    for feature_class in cfg.get("feature_classes", {}):
        extractor.enableFeatureClassByName(feature_class)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = extractor.execute(str(image_path), str(mask_path), label=label)
    diagnostics = {k: v for k, v in raw.items() if k.startswith("diagnostics_")}
    features = {}
    for key, value in raw.items():
        if key.startswith("diagnostics_"):
            continue
        if isinstance(value, np.ndarray):
            if value.size != 1:
                continue
            value = value.item()
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            features[key] = float(value)
    families = Counter(k.split("_")[1] if "_" in k else "unknown" for k in features)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "radiomics_features.json").write_text(json.dumps(features, indent=2, allow_nan=False), encoding="utf-8")
    (result_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    report = [
        "3D PyRadiomics extraction report", f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Input image: {image_path}", f"Input mask: {mask_path}",
        f"Image size xyz: {image.GetSize()}", f"Spacing xyz: {image.GetSpacing()}",
        f"ROI label: {label}", f"ROI voxels: {int(roi.sum())}",
        f"ROI physical volume mm3: {qc['roi']['physical_volume_mm3']}",
        f"Image intensity dtype: {image_array.dtype}",
        f"ROI intensity min/max/mean/std: {roi_values.min()} / {roi_values.max()} / {roi_values.mean():.6g} / {roi_values.std():.6g}",
        f"PyRadiomics: {radiomics.__version__}", f"SimpleITK: {sitk.Version_VersionString()}",
        f"Feature settings: {settings}", f"Image types: {cfg.get('image_types')}",
        f"Feature classes: {list(cfg.get('feature_classes', {}))}",
        f"Feature count: {len(features)}", f"Feature family counts: {dict(families)}",
        f"Warnings: {[str(w.message) for w in caught]}",
    ]
    (result_dir / "extraction_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"feature_count": len(features), "family_counts": dict(families), "output_dir": str(result_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("data/images/volume.nrrd"))
    parser.add_argument("--mask", type=Path, default=Path("data/masks/ROI_label.nrrd"))
    parser.add_argument("--config", type=Path, default=Path("config/radiomics.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    print(json.dumps(extract(args.image, args.mask, args.config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
