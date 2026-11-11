import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from scripts.extract_radiomics import extract


class RadiomicsTests(unittest.TestCase):
    def test_features_written_for_synthetic_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            z, y, x = np.indices((24, 24, 24))
            image_arr = (z * 2 + y + x * 3 + (x * y) % 7).astype(np.float32)
            mask_arr = np.zeros(image_arr.shape, dtype=np.uint8)
            mask_arr[5:19, 5:19, 5:19] = 1
            image = sitk.GetImageFromArray(image_arr)
            mask = sitk.GetImageFromArray(mask_arr)
            for obj in (image, mask):
                obj.SetSpacing((0.8, 0.8, 1.0))
                obj.SetMetaData("radiomics.spacing_calibrated", "true")
            image_path, mask_path = root / "image.nrrd", root / "mask.nrrd"
            sitk.WriteImage(image, str(image_path)); sitk.WriteImage(mask, str(mask_path))
            config = Path(__file__).parents[1] / "config" / "radiomics.yaml"
            result = extract(image_path, mask_path, config, root / "results")
            self.assertGreater(result["feature_count"], 0)
            self.assertTrue((root / "results" / "radiomics_features.json").is_file())
            self.assertTrue((root / "results" / "diagnostics.json").is_file())
            self.assertTrue((root / "results" / "extraction_report.txt").is_file())


if __name__ == "__main__":
    unittest.main()
