import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from scripts.mask_annotator import export_mask


class MaskTests(unittest.TestCase):
    def test_mask_export_copies_geometry_and_labels(self):
        reference = sitk.GetImageFromArray(np.zeros((7, 8, 9), dtype=np.float32))
        reference.SetSpacing((0.5, 0.7, 1.1))
        reference.SetOrigin((10.0, 20.0, 30.0))
        reference.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
        labels = np.zeros((7, 8, 9), dtype=np.uint8)
        labels[1:4, 2:6, 3:7] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.nrrd"
            saved = export_mask(labels, reference, path)
            self.assertEqual(saved.GetSize(), reference.GetSize())
            self.assertEqual(saved.GetSpacing(), reference.GetSpacing())
            self.assertEqual(saved.GetOrigin(), reference.GetOrigin())
            self.assertEqual(saved.GetDirection(), reference.GetDirection())
            self.assertTrue(np.array_equal(sitk.GetArrayFromImage(saved), labels))

    def test_non_integer_mask_rejected(self):
        reference = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                export_mask(np.zeros((2, 2, 2), dtype=np.float32), reference, Path(tmp) / "mask.nrrd")


if __name__ == "__main__":
    unittest.main()
