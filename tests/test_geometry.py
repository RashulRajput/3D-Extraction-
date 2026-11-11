import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from scripts.verify_geometry import verify
from scripts.quality_control import validate


def volume(data, spacing=(0.7, 0.7, 1.2), origin=(4.0, 5.0, 6.0)):
    image = sitk.GetImageFromArray(data)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetMetaData("radiomics.spacing_calibrated", "true")
    return image


class GeometryTests(unittest.TestCase):
    def test_matching_3d_mask_reports_roi_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.zeros((24, 24, 24), dtype=np.uint8)
            data[4:12, 5:13, 7:15] = 1
            image = volume(np.arange(data.size, dtype=np.float32).reshape(data.shape))
            mask = volume(data)
            sitk.WriteImage(image, str(root / "image.nrrd"))
            sitk.WriteImage(mask, str(root / "mask.nrrd"))
            report = verify(root / "image.nrrd", root / "mask.nrrd")
            self.assertEqual(report["geometry_check"], "PASS")
            self.assertEqual(report["roi"]["voxel_count"], 8 * 8 * 8)
            self.assertIsNotNone(report["roi"]["physical_volume_mm3"])

    def test_empty_mask_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = volume(np.ones((8, 8, 8), dtype=np.float32))
            mask = volume(np.zeros((8, 8, 8), dtype=np.uint8))
            sitk.WriteImage(image, str(root / "image.nrrd"))
            sitk.WriteImage(mask, str(root / "mask.nrrd"))
            report = verify(root / "image.nrrd", root / "mask.nrrd")
            self.assertEqual(report["geometry_check"], "FAIL")
            self.assertFalse(report["checks"]["mask_non_empty"])

    def test_geometry_mismatch_fails_without_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.ones((8, 8, 8), dtype=np.uint8)
            image, mask = volume(data), volume(data, origin=(4, 5, 7))
            sitk.WriteImage(image, str(root / "image.nrrd"))
            sitk.WriteImage(mask, str(root / "mask.nrrd"))
            report = verify(root / "image.nrrd", root / "mask.nrrd")
            self.assertEqual(report["geometry_check"], "FAIL")
            self.assertFalse(report["checks"]["origin_matches"])

    def test_missing_roi_label_reports_available_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.ones((8, 8, 8), dtype=np.uint8)
            image, mask = volume(data), volume(np.full(data.shape, 2, dtype=np.uint8))
            sitk.WriteImage(image, str(root / "image.nrrd"))
            sitk.WriteImage(mask, str(root / "mask.nrrd"))
            with self.assertRaisesRegex(ValueError, "available labels.*2"):
                validate(root / "image.nrrd", root / "mask.nrrd")


if __name__ == "__main__":
    unittest.main()
