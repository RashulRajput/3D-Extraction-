from pathlib import Path

try:
    from .verify_geometry import verify
except ImportError:
    from verify_geometry import verify


def validate(image_path=Path("data/images/volume.nrrd"), mask_path=Path("data/masks/ROI_label.nrrd"), roi_label=1):
    report = verify(Path(image_path), Path(mask_path), int(roi_label))
    if not report["checks"]["roi_label_present"]:
        raise ValueError(f"ROI label {roi_label} is absent; available labels: {report['roi']['available_labels']}")
    if report["geometry_check"] != "PASS":
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise ValueError("Geometry/mask QC failed: " + ", ".join(failed))
    if report["roi"]["voxel_count"] == 0:
        raise ValueError("ROI label 1 is empty.")
    return report
