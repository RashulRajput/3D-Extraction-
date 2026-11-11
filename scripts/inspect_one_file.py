import argparse
import json
from pathlib import Path

import pydicom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    ds = pydicom.dcmread(str(args.path), stop_before_pixels=True, force=False)
    tags = ("Modality", "StudyInstanceUID", "SeriesInstanceUID", "SOPClassUID", "SeriesDescription",
            "NumberOfFrames", "Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices",
            "ImagePositionPatient", "ImageOrientationPatient", "ImageType", "PhotometricInterpretation",
            "PixelRepresentation", "RescaleSlope", "RescaleIntercept")
    result = {k: str(getattr(ds, k)) for k in tags if hasattr(ds, k)}
    result["path"] = str(args.path)
    result["transfer_syntax"] = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "unknown"))
    result["per_frame_groups"] = len(getattr(ds, "PerFrameFunctionalGroupsSequence", []))
    result["shared_groups"] = len(getattr(ds, "SharedFunctionalGroupsSequence", []))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
