import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from scripts.convert_dicom import convert


class VolumeTests(unittest.TestCase):
    def test_nrrd_preserves_3d_data_and_geometry(self):
        array = np.arange(11 * 13 * 17, dtype=np.int16).reshape((11, 13, 17))
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.4, 0.6, 1.3))
        image.SetOrigin((2.0, 3.0, 4.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "volume.nrrd"
            sitk.WriteImage(image, str(path), True)
            loaded = sitk.ReadImage(str(path))
            self.assertEqual(loaded.GetSize(), image.GetSize())
            self.assertEqual(loaded.GetSpacing(), image.GetSpacing())
            self.assertEqual(loaded.GetOrigin(), image.GetOrigin())
            self.assertEqual(loaded.GetDirection(), image.GetDirection())
            self.assertTrue(np.array_equal(sitk.GetArrayFromImage(loaded), array))

    def test_synthetic_spatial_us_series_converts_with_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study_uid, series_uid = generate_uid(), generate_uid()
            paths = []
            for z in range(3):
                path = root / f"slice{z}"
                meta = FileMetaDataset()
                meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"
                meta.MediaStorageSOPInstanceUID = generate_uid()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
                ds.is_little_endian = True
                ds.is_implicit_VR = False
                ds.SOPClassUID = meta.MediaStorageSOPClassUID
                ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                ds.StudyInstanceUID = study_uid
                ds.SeriesInstanceUID = series_uid
                ds.Modality = "US"
                ds.Rows = 8
                ds.Columns = 8
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated = 8
                ds.BitsStored = 8
                ds.HighBit = 7
                ds.PixelRepresentation = 0
                ds.PixelSpacing = [0.8, 0.8]
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.ImagePositionPatient = [0, 0, z * 1.2]
                ds.InstanceNumber = z + 1
                ds.PixelData = np.full((8, 8), z + 1, dtype=np.uint8).tobytes()
                ds.save_as(path, write_like_original=False)
                paths.append(path)
            output = root / "volume.nrrd"
            image, report = convert(paths, output)
            self.assertEqual(image.GetDimension(), 3)
            self.assertEqual(image.GetSize(), (8, 8, 3))
            self.assertEqual(image.GetSpacing(), (0.8, 0.8, 1.2))
            self.assertEqual(report["spatial_positions_detected"], 3)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
