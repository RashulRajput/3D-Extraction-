import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydicom.multival import MultiValue

from scripts.inspect_dicomdir import _ref_paths, _spatial_status


class DicomDirTests(unittest.TestCase):
    def test_referenced_file_id_parts_resolve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = root / "DICOMOBJ" / "00000000"
            obj.parent.mkdir()
            obj.touch()
            ds = SimpleNamespace(DirectoryRecordSequence=[SimpleNamespace(ReferencedFileID=MultiValue(str, ["DICOMOBJ", "00000000"]))])
            self.assertEqual(_ref_paths(ds, root), [obj])

    def test_cine_is_not_spatial_3d_without_positions(self):
        status, _ = _spatial_status([{"NumberOfFrames": "24", "Rows": "512", "Columns": "512"}])
        self.assertEqual(status, "cine_or_unverified_multiframe")

    def test_single_frame_is_not_spatial_3d(self):
        status, _ = _spatial_status([{"Rows": "512", "Columns": "512"}])
        self.assertEqual(status, "single_2d_or_insufficient_geometry")

    def test_irregular_slice_gaps_are_not_a_3d_candidate(self):
        base = {"ImageOrientationPatient": ["1", "0", "0", "0", "1", "0"], "Rows": "512", "Columns": "512"}
        items = [{**base, "ImagePositionPatient": ["0", "0", str(z)]} for z in (1360.4, 1531.9, 1732.1)]
        status, _ = _spatial_status(items)
        self.assertEqual(status, "spatial_geometry_unverified")

    def test_regular_spatial_slice_gaps_are_candidate(self):
        base = {"ImageOrientationPatient": ["1", "0", "0", "0", "1", "0"], "Rows": "16", "Columns": "16"}
        items = [{**base, "ImagePositionPatient": ["0", "0", str(z)]} for z in (0, 1.2, 2.4)]
        status, _ = _spatial_status(items)
        self.assertEqual(status, "spatially_ordered_slices_candidate")


if __name__ == "__main__":
    unittest.main()
